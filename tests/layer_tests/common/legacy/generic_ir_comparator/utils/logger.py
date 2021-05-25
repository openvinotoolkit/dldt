import logging as log
import re


class TagFilter(log.Filter):
    def __init__(self, regex=None):
        log.Filter.__init__(self)
        self.regex = regex

    def filter(self, record):
        if record.__dict__['funcName'] == 'load_grammar':  # for nx not to log into our logs
            return False
        if self.regex:
            if 'tag' in record.__dict__.keys():
                tag = record.__dict__['tag']
                return re.findall(self.regex, tag)
            else:
                return False
        else:  # if regex wasn't set print all logs
            return True


def init_logger(lvl):
    logger = log.getLogger(__name__)
    log.basicConfig(
        format='%(levelname)s:%(message)s',
        level=lvl
    )
    logger.addFilter(TagFilter())
