import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.controllers.hello_controller import router as hello_router
from app.controllers.student_controller import router as student_router
from app.controllers.user_controller import router as user_router
from app.controllers.backgroundtasks_controller import router as background_tasks_router
from pydantic import ValidationError
from app.exceptions.handlers import (
    http_exception_handler,
    validation_exception_handler,
)
from app.exceptions.student_exceptions import StudentAgeException
from app.controllers.order_controller import router as order_router
from dotenv import load_dotenv # Used to load configuration from .env files
from app.controllers.oop_demo_controller import router as oop_router

# ----------------------------------------------------------------------
# DYNAMIC ENVIRONMENT LOADING LOGIC
# ----------------------------------------------------------------------

# 1a. Determine the environment to load. 
# It checks the shell's APP_ENV variable (e.g., set via $env:APP_ENV="prod").
# It defaults to 'local' if APP_ENV is not set.
app_env = os.environ.get("APP_ENV", "local") 
env_file_path = f".env.{app_env}"

# 1b. Load the specified environment file.
try:
    # Attempt to load the environment-specific file (e.g., .env.dev)
    load_dotenv(dotenv_path=env_file_path)
    print(f"--- SUCCESS: Configuration loaded from {env_file_path} ---")
except FileNotFoundError:
    # If the specific file is missing, fall back to the standard .env
    load_dotenv() 
    print(f"--- WARNING: Specific config file '{env_file_path}' not found. Loaded default .env ---")
except Exception as e:
    print(f"--- ERROR loading environment file: {e} ---")

# ----------------------------------------------------------------------
# APPLICATION INITIALIZATION & CONFIGURATION
# ----------------------------------------------------------------------

# Now, we can safely access loaded environment variables for configuration.
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
APP_TITLE = os.getenv("APP_TITLE", "Layered Python API")

# ----------------------------------------------------------------------

# 1. Initialize the FastAPI application.
# This creates the main application instance. Think of this as defining the Spring container.
# Arguments (title, version, description): These are crucial because FastAPI uses them to automatically generate your OpenAPI documentation (Swagger UI), which you can access at /docs.
app = FastAPI(
    title="Layered Python API",
    version="0.1.0",
    description="A multi-layered (Controller, Service, Repository) REST API."
)

# 2. Include the router from your controller
# The /api/v1 prefix is a common practice for versioning APIs.
app.include_router(hello_router, prefix="/api/v1")
app.include_router(student_router)
app.include_router(user_router)
app.include_router(order_router)
app.include_router(background_tasks_router)
app.include_router(oop_router)

# Register global handlers (ControllerAdvice equivalent)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)

# 3. Add a simple root endpoint for health checks (optional)
@app.get("/", tags=["Health"])
async def root():
    return {"message": "API is running. Access endpoints via /api/v1"}

# Note: In a real app, you would also set up database connections here
# and use them as dependencies in your controllers/services.

# ------------------------------------------------------------------------------------------

# This file is the heart of our FastAPI application! 
# This main.py serves the same purpose as the main class in a Spring Boot application annotated with @SpringBootApplication.
# It sets up the framework, defines global settings, and wires all your different components (controllers) together.

# from app.controllers.hello_controller import router as hello_router
# Explanation: 
# from app.controllers.hello_controller: This is standard Python module navigation, telling the application to look inside the app directory, 
#                                        then the controllers directory, and find the hello_controller.py file.
# import router: Inside your hello_controller.py, you defined an instance of APIRouter named router (e.g., router = APIRouter()).
#                This line pulls that specific object into main.py.
# as hello_router: This is a Python alias. It's a best practice to rename the imported router object (especially when you have multiple controllers, 
#                  e.g., user_router, item_router) to prevent naming conflicts and clearly identify where it came from.


# ------------------------------------------------------------------------------------------

# app.include_router(hello_router, prefix="/api/v1")
# Explanation: 
# app.include_router(hello_router): This is the crucial registration step. You are telling the main FastAPI application instance (app) to take all the endpoints defined 
#                                   in the hello_router and make them accessible.
# prefix="/api/v1": This is extremely important for modern APIs. It means:
#                   If your controller defined an endpoint as /hello.
#                   The final, publicly accessible path becomes /api/v1/hello.
#                   This allows you to easily introduce future versions (e.g., /api/v2) without breaking old client integrations.
#
# Analogy to Spring Boot: This is equivalent to setting a global path prefix on a Spring @RestController class using an annotation like: @RequestMapping("/api/v1/hello").


# ---------------------------------------------------------------------------------------------------

# Global exception handler for custom exception
@app.exception_handler(StudentAgeException)
async def student_age_exception_handler(request: Request, exc: StudentAgeException):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": exc.message,
            "submitted_age": exc.age,
            "path": str(request.url)
        }
    )

# Why we require the above Global Exception Handler ?
# FastAPI does not know how to automatically convert your custom exception into an HTTP response.
# Without a handler, the API will return a 500 Internal Server Error by default.
# ✅ Essentially, this is FastAPI’s version of Spring Boot’s @ControllerAdvice, giving consistent, clean error responses across all routes.
# Without the handler: FastAPI returns 500 Internal Server Error.