import warnings

try:
    from .parse import main
except ImportError as e:
    warnings.warn("Did you make sure to install PyProf dependencies by using the --pyprof flag during Apex installation?)")
    raise e

if __name__ == '__main__':
    main()
