import csv


class Logger:
    """Documentation for Logger
    filepath: path of the file to log data
    array: 2D array containing in each row the epoch stats
    """

    def __init__(self, filepath):
        super(Logger, self).__init__()
        self.filepath = filepath
        self.is_open = False
        self.f = open(self.filepath, 'w')
        self.f.close()
        self.log_buffer = []

    def open(self):
        if self.is_open:
            return
        self.f = open(self.filepath, 'a')
        self.is_open = True

    def close(self):
        if not self.is_open:
            return
        self.f.close()
        self.is_open = False

    def __write__(self):
        for data in self.log_buffer:
            self.f.write("\n")
            self.f.write(data)

    def log(self, data, show_output=True):
        self.log_buffer.append(data)
        if show_output:
            print(data)

    def save_log(self):
        self.open()
        self.__write__()
        self.log_buffer = []
        self.close()


class LoggerCSV(Logger):
    """Documentation for Logger
    filepath: path of the file to log data
    array: 2D array containing in each row the epoch stats

    """

    def __init__(self, filepath):
        super().__init__(filepath)

    def __write__(self):
        writer = csv.writer(self.f)
        for data in self.log_buffer:
            writer.writerow(data)
