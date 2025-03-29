from pydantic import BaseModel
from datetime import datetime
import bcrypt

class User(BaseModel):
    email: str
    username: str = None
    password: str
    created_at: int = int(datetime.timestamp(datetime.now()))  

    def hash_password(self):
        """Hash the user's password."""
        self.password = bcrypt.hashpw(self.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify the provided password against the stored hashed password."""
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
