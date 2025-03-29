def individual_data(user):
    return {
        "id":str(user["_id"]),
        "email":user["email"],
        "username":user["username"],
        "password":user["password"]
    }

def all_data(users):
    return{
        "users": [individual_data(user) for user in users]
    }
    