import asyncio
import os
import sys

# Add src/backend/base to sys.path
sys.path.append(r'd:\Work\langflow\langflow\src\backend\base')
sys.path.append(r'd:\Work\langflow\langflow\src\lfx\src')

from langflow.services.deps import get_variable_service, session_scope

async def test_get_all():
    service = get_variable_service()
    async with session_scope() as session:
        # Get user ID (assuming first user)
        from langflow.services.database.models.user.model import User
        from sqlmodel import select
        user = (await session.exec(select(User))).first()
        if not user:
            print("No user found")
            return
            
        print(f"Testing for user: {user.username} ({user.id})")
        variables = await service.get_all(user.id, session=session)
        print(f"Total variables returned by get_all: {len(variables)}")
        for i, var in enumerate(variables):
            print(f"{i+1}. {var.name} (Type: {var.type}, Value: {var.value})")

if __name__ == "__main__":
    asyncio.run(test_get_all())
