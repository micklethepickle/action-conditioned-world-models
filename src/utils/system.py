import datetime
import dateutil.tz



def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime(
        "%Y-%m-%d-%H:%M:%S"
    )  # may cause collision, please use PID to prevent
