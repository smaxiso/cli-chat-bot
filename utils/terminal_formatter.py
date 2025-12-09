from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)


class TerminalFormatter:
    """Handles terminal formatting and colors."""

    # Color constants
    API_NAME_COLOR = Fore.CYAN + Style.BRIGHT
    QUERY_COLOR = Fore.GREEN
    RESPONSE_HEADER_COLOR = Fore.YELLOW + Style.BRIGHT
    RESPONSE_TEXT_COLOR = Fore.WHITE
    SYSTEM_MSG_COLOR = Fore.MAGENTA
    ERROR_COLOR = Fore.RED
    HELP_COLOR = Fore.BLUE
    DIVIDER_COLOR = Fore.YELLOW
    PROMPT_COLOR = Fore.CYAN

    @staticmethod
    def format_api_name(api_name):
        """Format API name with color."""
        return f"{TerminalFormatter.API_NAME_COLOR}[{api_name.capitalize()}]{Style.RESET_ALL}"

    @staticmethod
    def format_query(query):
        """Format user query with color."""
        return f"{TerminalFormatter.QUERY_COLOR}{query}{Style.RESET_ALL}"

    @staticmethod
    def format_response(response, api_name):
        """Format AI response with color and styling."""
        divider = TerminalFormatter.DIVIDER_COLOR + "=" * 70 + Style.RESET_ALL
        header = f"{TerminalFormatter.RESPONSE_HEADER_COLOR}{api_name.capitalize()} Response:{Style.RESET_ALL}\n"
        formatted_text = f"{TerminalFormatter.RESPONSE_TEXT_COLOR}{response}{Style.RESET_ALL}"

        return f"\n{divider}\n{header}{formatted_text}\n{divider}\n"

    @staticmethod
    def format_system_message(message):
        """Format system messages with color."""
        return f"{TerminalFormatter.SYSTEM_MSG_COLOR}{message}{Style.RESET_ALL}"

    @staticmethod
    def format_error(message):
        """Format error messages with color."""
        return f"{TerminalFormatter.ERROR_COLOR}Error: {message}{Style.RESET_ALL}"

    @staticmethod
    def format_help_text(help_text):
        """Format help text with color."""
        return f"{TerminalFormatter.HELP_COLOR}{help_text}{Style.RESET_ALL}"

    @staticmethod
    def format_prompt(prompt):
        """Format input prompt with color."""
        return f"{TerminalFormatter.PROMPT_COLOR}{prompt}{Style.RESET_ALL}"
