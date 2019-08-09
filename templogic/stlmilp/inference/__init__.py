import logging

logger = logging.getLogger('lltinf')
logger.addHandler(logging.NullHandler())

import sys
if 'nose' in sys.modules.keys():
    import logging.config
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'debug_formatter': {
                'format': '%(levelname).1s %(module)s:%(lineno)d:%(funcName)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level':'DEBUG',
                'class':'logging.StreamHandler',
                'formatter':'debug_formatter',
            },
        },
        'loggers': {
            'lltinf': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })
