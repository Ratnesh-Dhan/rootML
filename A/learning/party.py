class Party:
    def __init__(self, venue):
        self.members = []
        self.host = ""
        self.venue = venue
        self.date = ""
        self.time = ""
        self.description = ""
    
    def get_date(self):
        return self.date

    def set_date(self, date):
        self.date = date

    # Getter and Setter for members
    def get_members(self):
        return self.members

    def set_members(self, members):
        self.members = members

    # Getter and Setter for host
    def get_host(self):
        return self.host

    def set_host(self, host):
        self.host = host

    # Getter and Setter for venue
    def get_venue(self):
        return self.venue

    def set_venue(self, venue):
        self.venue = venue

    # Getter and Setter for time
    def get_time(self):
        return self.time

    def set_time(self, time):
        self.time = time

    # Getter and Setter for description
    def get_description(self):
        return self.description

    def set_description(self, description):
        self.description = description