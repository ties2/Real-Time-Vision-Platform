class BBAPException(Exception):
    """Base exception for application-level errors."""

    code = "APPLICATION_ERROR"
    status_code = 500

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelNotFoundError(BBAPException):
    """Raised when a requested model does not exist."""

    code = "MODEL_NOT_FOUND"
    status_code = 404


class InvalidInputError(BBAPException):
    """Raised when inference input is invalid."""

    code = "INVALID_INPUT"
    status_code = 400
