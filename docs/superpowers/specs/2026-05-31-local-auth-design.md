## Local Auth Design

Goal: replace the missing Spring Boot auth endpoints with the minimum local FastAPI implementation while preserving the JWT shape the current AI service expects.

Scope:
- Add `POST /api/auth/signup`
- Add `POST /api/auth/signin`
- Store users in MongoDB database `budget-tracker`, collection `users`
- Return a Spring-like JWT containing `sub`, `accountId`, `email`, `roles`, `iat`, and `exp`

Architecture:
- `routes/auth.py` will expose the endpoints.
- `auth_service.py` will hold Mongo access, password hashing, and JWT generation.
- `models/schemas.py` will define request/response models.
- `routes/__init__.py` and `main.py` will register the router.

Data model:
- `_id`: Mongo object id
- `userId`: UUID used as JWT `sub`
- `accountId`: UUID claim preserved from the old token shape
- `email`: unique login identifier
- `passwordHash`: derived local password hash
- `roles`: string, default `ROLE_USER`
- `createdAt`: UTC timestamp

Behavior:
- Signup rejects duplicate email, creates a user, and returns token plus basic user info.
- Signin verifies email/password and returns token plus the same basic user info.
- JWT signing uses HS256 with `JWT_SECRET` from environment, falling back to a local-dev default if unset.
- Token expiry defaults to 7 days.

Error handling:
- `400` for duplicate email
- `401` for invalid credentials
- `500` for unexpected Mongo or encoding failures

Testing:
- Unit-test the route functions with a fake collection/service boundary.
- Verify signup success, duplicate email handling, signin success, and signin failure.
