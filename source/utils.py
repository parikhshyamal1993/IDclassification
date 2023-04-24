import configparser , os

config = configparser.ConfigParser()
config.read("./config/config.ini")

def ConfigParser():
    configDict = {}
    print(config.options('DEV'))
    for items  in config.options('DEV'):
        configDict[items] = config.get('DEV',items)
    return configDict
Config = ConfigParser()

def FileHandler(Filename):

    extenstions = Config["filetypes"].split(",")
    for Ftypes in extenstions:
        if Ftypes in Filename:
            return True
    else: 
        return False




if __name__=="__main__":
    print(FileHandler("file.png"))