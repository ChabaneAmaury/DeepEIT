import csv
from os import makedirs
from os.path import exists, dirname

from pymongo import MongoClient


class MongodbAPI:
    def __init__(self, client='localhost', port=27017, db=None, collection=None):
        """
        This class contains every tools we need to interact with the mongoDB server.
        """
        self.client = MongoClient(client, port)
        self.db = None
        self.collection = None
        if db is not None:
            self.setDB(db)
        if db is not None and collection is not None:
            self.setCollection(collection)

    def setDB(self, db):
        """
        Sets the database to work on.

        :param db: (str) The database's name
        :return: None
        """
        self.db = self.client[db]

    def setCollection(self, collection):
        """
        Sets the collection to work with.

        :param collection: (str) The collection's name
        :return: None
        """

        if collection in self.db.list_collection_names():
            self.collection = self.db[collection]
        else:
            self.collection = self.createCollection(collection)

    def insertData(self, data, collection, drop=False):
        """
        Insert the given list inside the given collection.

        :param collection: (str) The collection's name
        :param data: (list) the data to insert, in form of a list
        :param drop: (Boolean) Define weather or not we drop the collection
        :return: (int) The new number of elements in the collection
        """
        self.setCollection(collection)
        if drop:
            self.collection.drop()
            self.createCollection(collection)

        self.collection.insert_many(data)
        return self.collection.count_documents({})

    def createCollection(self, collection):
        """
        Create the defined collection with a higher compression method.

        :param collection: (str) The compression's name
        :return:
        """
        self.db.create_collection(collection,
                                  storageEngine={'wiredTiger': {'configString': 'block_compressor=zstd'}})
        return self.db[collection]

    def findAllData(self, collection, limit: int = 0):
        """
        :param collection:(str) The collection's name
        :param limit: (int) The limit of each iteration to fetch the data
        :return: (list) The result of the collection.find() query. Either a full list or a generator,
        depending on the limit's value
        """
        skip = 0
        if limit > 0:
            total = self.db[collection].count_documents({})
            print(total)
            while skip != total:
                data = list(self.db[collection].find().skip(skip).limit(limit))
                skip += len(data)
                yield data
        else:
            return list(self.db[collection].find())

    def export_collection_to_csv(self, path):
        if self.collection.count_documents({}) == 0:
            return

        fieldnames = list(self.collection.find_one().keys())
        fieldnames.remove('_id')

        mongo_docs = self.collection.find({})

        if not exists(path):
            base = dirname(path)
            makedirs(base)

        with open(path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(mongo_docs)
