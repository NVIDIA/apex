# This is a "header object" that allows different amp modules to communicate.
# I'm a C++ guy, not a python guy.  I decided this approach because it seemed most C++-like.  
# But apparently it's ok:
# http://effbot.org/pyfaq/how-do-i-share-global-variables-across-modules.htm
class AmpState(object):
    def __init__(self):
        self.hard_override=False

# Attribute stash.  Could also just stash things as global module attributes.
_amp_state = AmpState()

def warn_or_err(msg):
    if _amp_state.hard_override:
        print("Warning:  " + msg)
    else:
        raise RuntimeError(msg + "  If you're sure you know what you're doing, supply " +
                           "hard_override=True to amp.initialize.")
