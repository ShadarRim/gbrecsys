import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:

    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
    
        # нам потребуется топ покупок для каждого user
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        #не забываем удалять 999999, иначе всё не слава богу
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]
    
        # нам потребуется список всех покупок
        self.all_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.all_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        # так как мы создавали 999999, то его надо удалить, иначе всё не слава богу
        self.all_top_purchases = self.all_top_purchases[self.all_top_purchases['item_id'] != 999999]
        self.all_top_purchases = self.all_top_purchases.item_id.tolist()
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        
        # your_code
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_N_u_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_N_u_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)
        
        if len(res) < N:
            #придётся дополнитеь список, чтобы всегда иметь одинаковое количество товаров
            #будет добавлять товары до нужно размера из общего рейтинга покупок, который для этго придётся делать
            #можно и случайно, конечно, но кажется, что так точнее
            res.extend(self.all_top_purchases[:N])
            res = res[:N]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
    
    #"""Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        res = []

       
        sim_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        sim_users = [rec[0] for rec in sim_users]
        sim_users = sim_users[1:]   # сам user тоже найдётся как похожий

        for user in sim_users:
            res.extend(self.get_own_recommendations(user, N=1))


        if len(res) < N:
            #придётся дополнитеь список, чтобы всегда иметь одинаковое количество товаров
            #будет добавлять товары до нужно размера из общего рейтинга покупок, который для этго придётся делать
            #можно и случайно, конечно, но кажется, что так точнее
            res.extend(self.all_top_purchases[:N])
            res = res[:N]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res