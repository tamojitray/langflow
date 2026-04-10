from typing import Any

from lfx.base.models.unified_models import (
    apply_provider_variable_config_to_build_config,
    clear_provider_specific_fields,
    get_language_model_options,
    get_llm,
    get_provider_for_model_name,
    update_model_options_in_build_config,
)
from lfx.custom import Component
from lfx.io import (
    BoolInput,
    MessageInput,
    MessageTextInput,
    ModelInput,
    MultilineInput,
    Output,
    SecretStrInput,
    TableInput,
)
from lfx.schema.message import Message
from lfx.schema.table import EditMode


class SmartRouterComponent(Component):
    display_name = "Smart Router"
    description = "Routes an input message using LLM-based categorization."
    icon = "route"
    name = "SmartRouter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._matched_category = None
        self._categorization_result: str | None = None

    inputs = [
        ModelInput(
            name="model",
            display_name="Language Model",
            info="Select your model provider",
            real_time_refresh=True,
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Model Provider API key",
            advanced=True,
        ),
        MessageTextInput(
            name="azure_endpoint",
            display_name="Azure OpenAI Endpoint",
            info="The endpoint URL of the Azure OpenAI service (Azure OpenAI only)",
            show=False,
            advanced=True,
        ),
        MessageTextInput(
            name="azure_deployment",
            display_name="Azure OpenAI Deployment",
            info="The deployment name for the Azure OpenAI service (Azure OpenAI only)",
            show=False,
            advanced=True,
        ),
        MessageTextInput(
            name="input_text",
            display_name="Input",
            info="The primary text input for the operation.",
            required=True,
        ),
        TableInput(
            name="routes",
            display_name="Routes",
            info=(
                "Define the categories for routing. Each row should have a route/category name "
                "and optionally a custom output value."
            ),
            table_schema=[
                {
                    "name": "route_category",
                    "display_name": "Route Name",
                    "type": "str",
                    "description": "Name for the route (used for both output name and category matching)",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "route_description",
                    "display_name": "Route Description",
                    "type": "str",
                    "description": "Description of when this route should be used (helps LLM understand the category)",
                    "default": "",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "output_value",
                    "display_name": "Route Message (Optional)",
                    "type": "str",
                    "description": (
                        "Optional message to send when this route is matched."
                        "Leave empty to pass through the original input text."
                    ),
                    "default": "",
                    "edit_mode": EditMode.POPOVER,
                },
            ],
            value=[
                {
                    "route_category": "Positive",
                    "route_description": "Positive feedback, satisfaction, or compliments",
                    "output_value": "",
                },
                {
                    "route_category": "Negative",
                    "route_description": "Complaints, issues, or dissatisfaction",
                    "output_value": "",
                },
            ],
            required=True,
        ),
        MessageInput(
            name="message",
            display_name="Override Output",
            info=(
                "Optional override message that will replace both the Input and Output Value "
                "for all routes when filled."
            ),
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="enable_else_output",
            display_name="Include Else Output",
            info="Include an Else output for cases that don't match any route.",
            value=False,
            advanced=True,
        ),
        MultilineInput(
            name="custom_prompt",
            display_name="Additional Instructions",
            info=(
                "Additional instructions for LLM-based categorization. "
                "These will be added to the base prompt. "
                "Use {input_text} for the input text and {routes} for the available categories."
            ),
            advanced=True,
        ),
    ]

    outputs: list[Output] = []

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
        """Dynamically update build config with user-filtered model options."""
        # Only update model options if relevant to prevent infinite loops on selection
        if field_name == "model" or not build_config.get("model", {}).get("options"):
            build_config = update_model_options_in_build_config(
                component=self,
                build_config=build_config,
                cache_key_prefix="language_model_options",
                get_options_func=get_language_model_options,
                field_name=field_name,
                field_value=field_value,
            )

        current_model_value = field_value if field_name == "model" else build_config.get("model", {}).get("value")
        provider = ""

        # Improved provider detection to be more resilient during reload
        if isinstance(current_model_value, list) and current_model_value:
            selected_model = current_model_value[0]
            provider = (selected_model.get("provider") or "").strip()
            if not provider and selected_model.get("name"):
                provider = get_provider_for_model_name(str(selected_model["name"]))
        elif isinstance(current_model_value, dict):
            provider = (current_model_value.get("provider") or "").strip()
            if not provider and current_model_value.get("name"):
                provider = get_provider_for_model_name(str(current_model_value["name"]))
        elif isinstance(current_model_value, str):
            provider = get_provider_for_model_name(current_model_value)

        if provider:
            build_config = apply_provider_variable_config_to_build_config(build_config, provider)
        elif field_name == "model":
            # Only clear if the user explicitly changed the model to something invalid
            build_config = clear_provider_specific_fields(build_config)

        return build_config

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Create a dynamic output for each category in the categories table."""
        # Get existing outputs from node template
        old_outputs_list = frontend_node.get("outputs", [])

        # If this is a generic update (selected or refresh) and we already have outputs, return early.
        # This breaks the selection loop.
        if field_name not in {"routes", "enable_else_output"} and old_outputs_list:
            return frontend_node

        # Preserve existing outputs state (dict of old outputs)
        old_outputs = {out.get("name"): out for out in old_outputs_list}

        # Get the routes data - prioritize what's in the node template on first reload
        routes_data = field_value if field_name == "routes" else getattr(self, "routes", [])
        if not routes_data and "template" in frontend_node:
            routes_data = frontend_node["template"].get("routes", {}).get("value", [])

        # If we still have no routes, we use the default class outputs if any
        if not routes_data:
            return frontend_node

        new_outputs = []
        # Add a dynamic output for each category
        for i, row in enumerate(routes_data):
            route_category = row.get("route_category", f"Category {i + 1}")
            output_name = f"category_{i + 1}_result"

            new_output = Output(
                display_name=route_category,
                name=output_name,
                method="process_case",
                group_outputs=True,
            )

            # Restore state if output already existed (selected, hidden, value, etc.)
            if output_name in old_outputs:
                old_out = old_outputs[output_name]
                for attr in ["selected", "hidden", "value", "types", "cache"]:
                    if attr in old_out:
                        setattr(new_output, attr, old_out[attr])

            new_outputs.append(new_output)

        # Add default output only if enabled
        is_else_enabled = (
            field_value if field_name == "enable_else_output" else getattr(self, "enable_else_output", False)
        )
        if is_else_enabled:
            else_output_name = "default_result"
            new_else_output = Output(
                display_name="Else",
                name=else_output_name,
                method="default_response",
                group_outputs=True,
            )

            if else_output_name in old_outputs:
                old_else = old_outputs[else_output_name]
                for attr in ["selected", "hidden", "value", "types", "cache"]:
                    if attr in old_else:
                        setattr(new_else_output, attr, old_else[attr])

            new_outputs.append(new_else_output)

        # Structural comparison to prevent loops
        new_structural = [{"name": out.name, "display_name": out.display_name, "method": out.method} for out in new_outputs]
        old_structural = [
            {"name": out.get("name"), "display_name": out.get("display_name"), "method": out.get("method")}
            for out in old_outputs_list
        ]

        if new_structural != old_structural:
            # Only update node if there's a structural change
            frontend_node["outputs"] = [out.to_dict() if hasattr(out, "to_dict") else out for out in new_outputs]

        return frontend_node

    def _get_categorization(self) -> str:
        """Perform LLM categorization and cache the result.

        This ensures the LLM is called only once per component execution,
        regardless of how many outputs are connected.
        """
        # Return cached result if available
        if self._categorization_result is not None:
            return self._categorization_result

        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")
        llm = get_llm(
            model=self.model,
            user_id=self.user_id,
            api_key=self.api_key,
            azure_endpoint=getattr(self, "azure_endpoint", None),
            azure_deployment=getattr(self, "azure_deployment", None),
        )

        if not llm or not categories:
            self.status = "No LLM provided for categorization"
            self._categorization_result = "NONE"
            return self._categorization_result

        # Create prompt for categorization
        category_info = []
        for i, category in enumerate(categories):
            cat_name = category.get("route_category", f"Category {i + 1}")
            cat_desc = category.get("route_description", "")
            if cat_desc and cat_desc.strip():
                category_info.append(f'"{cat_name}": {cat_desc}')
            else:
                category_info.append(f'"{cat_name}"')

        categories_text = "\n".join([f"- {info}" for info in category_info if info])

        # Create base prompt
        base_prompt = (
            f"You are a text classifier. Given the following text and categories, "
            f"determine which category best matches the text.\n\n"
            f'Text to classify: "{input_text}"\n\n'
            f"Available categories:\n{categories_text}\n\n"
            f"Respond with ONLY the exact category name that best matches the text. "
            f'If none match well, respond with "NONE".\n\n'
            f"Category:"
        )

        # Use custom prompt as additional instructions if provided
        custom_prompt = getattr(self, "custom_prompt", "")
        if custom_prompt and custom_prompt.strip():
            self.status = "Using custom prompt as additional instructions"
            simple_routes = ", ".join(
                [f'"{cat.get("route_category", f"Category {i + 1}")}"' for i, cat in enumerate(categories)]
            )
            formatted_custom = custom_prompt.format(input_text=input_text, routes=simple_routes)
            prompt = f"{base_prompt}\n\nAdditional Instructions:\n{formatted_custom}"
        else:
            self.status = "Using default prompt for LLM categorization"
            prompt = base_prompt

        self.status = f"Prompt sent to LLM:\n{prompt}"

        try:
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
                if hasattr(response, "content"):
                    categorization = response.content.strip().strip('"')
                else:
                    categorization = str(response).strip().strip('"')
            else:
                categorization = str(llm(prompt)).strip().strip('"')

            self.status = f"LLM response: '{categorization}'"
            self._categorization_result = categorization
        except RuntimeError as e:
            self.status = f"Error in LLM categorization: {e!s}"
            self._categorization_result = "NONE"

        return self._categorization_result

    def process_case(self) -> Message:
        """Process all categories using LLM categorization and return message for matching category."""
        # Clear any previous match state (only on first call)
        if self._categorization_result is None:
            self._matched_category = None

        # Get categories and input text
        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")

        # Get the cached categorization result (performs LLM call only once)
        categorization = self._get_categorization()

        # Find matching category based on LLM response
        matched_category = None
        for i, category in enumerate(categories):
            route_category = category.get("route_category", "")
            if categorization.lower() == route_category.lower():
                matched_category = i
                self.status = f"MATCH FOUND! Category {i + 1} matched with '{categorization}'"
                break

        if matched_category is not None:
            # Store the matched category for other outputs to check
            self._matched_category = matched_category

            # Stop all category outputs except the matched one
            for i in range(len(categories)):
                if i != matched_category:
                    self.stop(f"category_{i + 1}_result")

            # Also stop the default output (if it exists)
            enable_else = getattr(self, "enable_else_output", False)
            if enable_else:
                self.stop("default_result")

            route_category = categories[matched_category].get("route_category", f"Category {matched_category + 1}")
            self.status = f"Categorized as {route_category}"

            # Check if there's an override output (takes precedence over everything)
            override_output = getattr(self, "message", None)
            if (
                override_output
                and hasattr(override_output, "text")
                and override_output.text
                and str(override_output.text).strip()
            ):
                return Message(text=str(override_output.text))
            if override_output and isinstance(override_output, str) and override_output.strip():
                return Message(text=str(override_output))

            # Check if there's a custom output value for this category
            custom_output = categories[matched_category].get("output_value", "")
            # Treat None, empty string, or whitespace as blank
            if custom_output and str(custom_output).strip() and str(custom_output).strip().lower() != "none":
                # Use custom output value
                return Message(text=str(custom_output))
            # Use input as default output
            return Message(text=input_text)
        # No match found, stop all category outputs
        for i in range(len(categories)):
            self.stop(f"category_{i + 1}_result")

        # Check if else output is enabled
        enable_else = getattr(self, "enable_else_output", False)
        if enable_else:
            # The default_response will handle the else case
            self.stop("process_case")
            return Message(text="")
        # No else output, so no output at all
        self.status = "No match found and Else output is disabled"
        return Message(text="")

    def default_response(self) -> Message:
        """Handle the else case when no conditions match."""
        enable_else = getattr(self, "enable_else_output", False)
        if not enable_else:
            self.status = "Else output is disabled"
            return Message(text="")

        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")

        # Get the cached categorization result (performs LLM call only if not already done)
        categorization = self._get_categorization()

        # Check if the categorization matches any category
        has_match = False
        for i, category in enumerate(categories):
            route_category = category.get("route_category", "")
            if categorization.lower() == route_category.lower():
                has_match = True
                self.status = f"Match found for '{categorization}' (Category {i + 1}), stopping default_response"
                break

        if has_match:
            # A case matches, stop this output
            self.stop("default_result")
            return Message(text="")

        # No case matches, check for override output first, then use input as default
        override_output = getattr(self, "message", None)
        if (
            override_output
            and hasattr(override_output, "text")
            and override_output.text
            and str(override_output.text).strip()
        ):
            self.status = "Routed to Else (no match) - using override output"
            return Message(text=str(override_output.text))
        if override_output and isinstance(override_output, str) and override_output.strip():
            self.status = "Routed to Else (no match) - using override output"
            return Message(text=str(override_output))

        self.status = "Routed to Else (no match) - using input as default"
        return Message(text=input_text)
