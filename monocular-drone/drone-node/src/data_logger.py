
import json
import nav_config as cfg

class DataLogger(object):
    def __init__(self):
        self.data = dict()
        self.entry = dict()
        self.log = []

        self.data["cfg"] = cfg.get_configs()
        self.logfile = cfg.logfile

    def new_entry(self):
        if self.logfile is None:
            return

        self.entry = dict()

    def push_entry(self):
        if self.logfile is None:
            return

        if self.entry is not None and len(self.entry.keys()) > 0:
            self.log.append(self.entry)

    def add_key_val(self, key, val):
        if self.logfile is None:
            return

        if key is None or val is None:
            return 
        
        if not isinstance(key, str):
            return

        self.entry[key] = val

    def export_logs(self):
        if self.logfile is None:
            return

        self.data["log"] = self.log

        with open(self.logfile, "w") as json_file:
            print(f"Exporting {len(self.log)} entries to {self.logfile}")
            json.dump(self.data, json_file)  # Use `indent=4` for pretty formatting

        self.logfile = None

