"""Top-level API router registry."""

from fastapi import APIRouter

from app.api.routes.delete import DELETE_ROUTES
from app.api.routes.get import GET_ROUTES
from app.api.routes.post import POST_ROUTES
from app.api.routes.put import PUT_ROUTES


api_router = APIRouter(prefix="/api/v1")

ROUTES_BY_METHOD = {
    "GET": GET_ROUTES,
    "POST": POST_ROUTES,
    "PUT": PUT_ROUTES,
    "DELETE": DELETE_ROUTES,
}

for method, routes in ROUTES_BY_METHOD.items():
    for route in routes:
        api_router.add_api_route(
            path=route.path,
            endpoint=route.endpoint,
            methods=[method],
            name=route.name,
            response_model=route.response_model,
            status_code=route.status_code,
            tags=list(route.tags),
        )
