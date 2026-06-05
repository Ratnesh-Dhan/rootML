from party import Party    

if __name__ == "__main__":
    party = Party("Singh saab")

    party.set_date("07-06-2026")
    a = party.get_date()
    print(a)

    party.set_members(["Kanika", "Sai", "Priti", "Paddy", "Prabhakar da"])

    print(party.get_members())

    print(party.get_venue())

    party.set_time("05:00 PM")
    print(party.get_time())

    party.set_description("Party to celebrate the victory of the Indian team in the World Cup")
    print(party.get_description())

    party.set_host("Singh saab")
    print(party.get_host())
    
    