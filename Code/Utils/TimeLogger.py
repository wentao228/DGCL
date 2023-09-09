import datetime

logmsg = ''
saveDefault = False


def log(msg, save=None, oneline=False):
    """
    Log a message with an optional timestamp.

    Parameters:
    msg (str): The message to be logged.
    save (bool, optional): Whether to save the message. If None, it uses the default behavior defined by `saveDefault`.
    oneline (bool, optional): Whether to display the message on a single line.

    Returns:
    None
    """
    global logmsg
    global saveDefault
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if save != None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)


if __name__ == '__main__':
    log('')
