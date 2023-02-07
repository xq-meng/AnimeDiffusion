import logging

class MessageBox(logging.Handler):
    def __init__(self, text_browser)-> None:
        super().__init__()
        self.text_browser = text_browser 

    def emit(self, record) -> None:
        msg = self.format(record)
        self.text_browser.append(msg)