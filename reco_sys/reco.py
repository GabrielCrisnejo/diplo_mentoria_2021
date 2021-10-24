import numpy as np
from gensim.models import Word2Vec 
from gensim.utils import simple_preprocess
from gensim.utils import deaccent
from gensim.models import KeyedVectors
import random
import time

class RecoSys(object):
    """This class has as a main method the recommendation system
    and auxillary methods need for it to work as intended """

    def __init__(self, model, user_history, user_id, item_title): # add metadata later if we integrate the ndcg score
        """
        model           Word2Vec trained model
        user_history    Dictionary with user, user_view, user_search and bought_title 
        user_id         Integer to retrieve user_history, this should be in [0, len(user_history)]
        metadata        This should be a dictionary that has the title of each item viewd by the user
        item_title      A dictionary where the key is an item id and the value is the item title, this is preprocessed
        """
        

        self.model = model
        self.user_history = user_history

        if int(user_id) > len(self.user_history):
            print(f'User ID {user_id} out of bound of user_history, please initialize an User ID between 0 and {len(self.user_history)}')
        else:
            self.user_id = user_id
        
        # self.metadata = metadata
        self.user = self.user_history[self.user_id]
        self.item_title = item_title

        self.user_view = self.user['user_view']
        self.user_search = self.user['user_search']
        self.user_bought = self.user['bought_title']       

    def _preprocess(self):
        """
        This method preprocess the user view item title and bought title
        """
        # Read user view item titles
        user_view_title = [self.item_title[str(item_id)] for item_id in self.user_view]
        user_view_title_nvoid = [user_view_title[i] for i in range(len(user_view_title)) if user_view_title[i] != []]
        self.view = []
        for title in user_view_title_nvoid:
            titl = [word for word in title if word in self.model.wv]
            self.view.append(titl)
        
        self.view = [self.view[i] for i in range(len(self.view)) if self.view[i] != []]

        # Preprocess user search 
        search = [simple_preprocess(self.user_search[i]) for i in range(len(self.user_search))]
        self.search = [search[i] for i in range(len(search)) if search[i] != []]
    
    def model_ms(self):
        """
        This method will use wv.most_similar to find the top 10 words by search and view titles
        for the given user_id.
        """
        self._preprocess()

        self.recom_by_views = [self.model.wv.most_similar(view, topn=10)[i][0] for i in range(10) for view in self.view]
        self.recom_by_search = [self.model.wv.most_similar(search, topn=10)[i][0] for i in range(10) for search in self.search]
    
    def find_matches(self, min_match):
        """
        This method will find string matches in every item title and bought title 

        min_match          Minimum string matches
        """
        def allopt(recom, itemID):
            """
            This function finds matches of strings between two lists.

            recom       A 10 strings list
            itemID      ID of an item
            """
            return [elem in self.item_title[itemID] for elem in recom]
        
        self.model_ms()
        max_len = max(map(len, self.item_title))
        aux_view = {itemID : sum(allopt(self.recom_by_views, itemID))/max_len for itemID in self.item_title.keys() if sum(allopt(self.recom_by_views, itemID)) > min_match}
        aux_search = {itemID : sum(allopt(self.recom_by_search, itemID))/max_len for itemID in self.item_title.keys() if sum(allopt(self.recom_by_views, itemID)) > min_match}

        # Store top ten matches
        self._view_match = sorted(aux_view, key=aux_view.get, reverse=True)[:5]
        self._search_match = sorted(aux_search, key=aux_search.get, reverse=True)[:5]

        # The final 10-items-recommendation will be a list with the top 5 items of each match
        self.reco_list = self._view_match + self._search_match
      
  
class AutoReco(object):
  def __init__(self, max_len, min_match, model, user_history, item_title):
    self.max_len = max_len
    self.min_match = min_match
    self.model = model
    self.user_history = user_history
    self.item_title = item_title

  def reco_func(self):
    random.seed(23)
    sample = random.sample(range(0, len(self.user_history)), self.max_len)
    
    self.reco_list = {}
    
    total_start = time.time()
    i = 1
    for user in sample:
      start_loop = time.time()
      print(f'\rItem %d of {len(sample)} '%i, end="")
      reco_obj = RecoSys(self.model, self.user_history, str(user), self.item_title)
      reco_obj.find_matches(self.min_match)
      rl = reco_obj.reco_list
      self.reco_list[str(user)] = rl
      end_loop = time.time()
      total_time = np.round((end_loop - start_loop)/60,3)
      print(f'\rLoop %d took {total_time} minutes'%i, end="")
      i += 1
    
    total_end = time.time()
    total = np.round((total_end - total_start)/60, 3)
    print(f'{len(sample)} items took {total} minutes to find recommendation list')










