#!/user/bin/env python3
import json
import requests
import urllib.parse

class ArachneTable(object):
    def __init__(self, tblName, url, token):
        self.tblName = tblName
        self.__token = token
        self.__url = url

    def describe(self):
        '''
            returns col names of current table
        '''
        first_row = self.getAll(limit=1)
        if len(first_row) != 1: raise ValueError(f"error while retrieving columns from table '{self.tblName}'")
        return list(first_row[0].keys())

    def version(self):
        '''
            returns dict:
                "max_date" => datetime stamp of update
                "length" => number of rows in table
        '''
        rows = self.getAll(select=["u_date"], order=["u_date"])
        return {
            "max_date": rows[len(rows)-1]["u_date"],
            "length": len(rows)
        }
    
    def getAll(self, count=False, limit=0, offset=0, select=[], order=[], group=[]):
        return self.search([{"c":"id", "o":">", "v":0}], count, limit, offset, select, order, group)
    def get(self, query, count=False, limit=0, offset=0, select=[], order=[], group=[]):
        return self.search([{"c": item[0], "o": "=", "v": item[1]} for item in query.items()], count, limit, offset, select, order, group)
    def search(self, query, count=False, limit=0, offset=0, select=[], order=[], group=[]):
        """
            searches database:
                query:  search query in form of a list of dictionaries [{"c": c, "o": "v": v}, {"c": c, "o": "v": v}, ...]
                        where c = column; o = operator (=, !=, >=, <=, LIKE); v = value (str, int)
                        dictionaries will be connected by AND
                count:  returns only number of results (=COUNT(*)); BOOL
                limit:  limits number of results; INT
                offset: offset, only works if limit is set; INT
                select: returns given columns i.e. ["work", "author"] if empty all columns will be returned; LIST
                order:  order results by list of columns; LIST
                group:  groups results by list of columns; LIST
        """
        if not isinstance(query, list): raise ValueError("query must be a list!")
        params = {"query": json.dumps(query)}
        if count: params["count"] = 1
        if limit > 0: params["limit"] = limit
        if offset > 0:
            if limit == 0: raise ValueError("cannot set offset without setting limit!")
            else: params["offset"] = offset
        if len(select)>0: params["select"] = json.dumps(select)
        if len(order)>0: params["order"] = json.dumps(order)
        if len(group)>0: params["group"] = json.dumps(group)
        #url = f"{self.__url}/{self.tblName}?query={json.dumps(query)}"
        url = f"{self.__url}/data/{self.tblName}?"+urllib.parse.urlencode(params)
        #print("url:", url)
        re = requests.get(url, headers={"Authorization": f"Bearer {self.__token}"})
        if re.status_code == 200:
            return re.json()
        else:
            raise ConnectionRefusedError("cannot connect to database!")
    def delete(self, rowId):
            url = f"{self.__url}/data/{self.tblName}/{rowId}"
            data = None;
            if isinstance(rowId, list):
                url = f"{self.__url}/data_batch/${self.tblName}"
                data = json.dumps(rowId);
            re = requests.delete(
                url,
                headers={
                    "Authorization": f"Bearer {self.__token}",
                    "Content-Type": "application/json",
                    },
                json=data
                )
            if re.status_code==200:
                return True
            else:
                raise ConnectionRefusedError("cannot connect to database!")
    def save(self, newValues):
            # newValues is an object containing col/values as key/value pairs.
            # when no id is given, a new entry will be created.
            # for batch saving: newValues = [{col: val}, {col. val}, ...]
            method = "POST"
            url = ""
            rId = 1
            if isinstance(newValues, list):
                url = f"{self.__url}/data_batch/${self.tblName}"
            else:
                url = f"{self.__url}/data/{self.tblName}"
                rId = newValues.get("id", None)
                if newValues.get("id", None):
                    url += f"/{newValues['id']}"
                    method = "PATCH"
                    del newValues["id"]

            if method == "POST":
                re = requests.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.__token}"
                    },
                    json=newValues
                )
            elif method == "PATCH":
                re = requests.patch(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.__token}"
                    },
                    json=newValues
                )
            else: raise ValueError("unknown method.")
            
            if re.status_code==201 and method=="POST":
                if isinstance(newValues, list): return rId
                else: return int(re.text)
            elif re.status_code==200 and method=="PATCH":
                return rId
            elif isinstance(newValues, list):
                return True
            else:
                raise ConnectionRefusedError(f"cannot connect to database! status code: {re.status_code}")
class Arachne(object):
    def __init__(self, user, pw, url='https://dienste.badw.de:9999/mlw', tbls = None):
        self.__url = url
        re = requests.post(f"{self.__url}/session", json={"user": user, "password": pw})
        if re.status_code == 201:
            self.__token = re.text
        elif re.status_code == 401:
            raise PermissionError("Server is online but login credentials are refused!")
        else:
            raise ConnectionRefusedError(f"login failed. Check accessibility of url: {self.__url}")

        if tbls == None: raise ConnectionRefusedError("No tables specified! Check documentation for more information.")
        for tbl in tbls:
            setattr(self, tbl, ArachneTable(tbl, url, self.__token))
    def get_zettel_img(self, img_path):
        re = requests.get(f"{self.__url}{img_path}.jpg", headers={"Authorization": f"Bearer {self.__token}"})
        if re.status_code==200:
            return re.content
        else: raise ValueError(f"cannot download image '{self.__url}{img_path}.jpg'.")
    def close(self):
        ''' call close() at the end of the session to delete active token from server.'''
        re = requests.delete(f"{self.__url}/session", headers={"Authorization": f"Bearer {self.__token}"})
        if(re.status_code == 200):
            del self.__token
        else:
            raise ConnectionRefusedError("logout failed. Are you already logged out?")