#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import binomtest, mannwhitneyu


# In[55]:


likes = pd.read_csv('likes.csv')
likes.head()


# In[3]:


# пропущенных значений нет
# данные о лайках юзеров (когда, кем и какой пост лайкнут)
likes.info()


# In[4]:


likes.user_id.nunique(), likes.post_id.nunique()


# In[5]:


# число лайков на юзера - логнорм распределение
likes_for_users = likes.groupby('user_id', as_index = False).agg({'post_id' : 'count'}).rename(columns = {'post_id' : 'likes_count'})
likes_for_users.likes_count.hist()


# In[56]:


# данные о просмотрах
views = pd.read_csv('views.csv')
views.head()


# In[9]:


# без пропущенных значений
# данные о просмотрах рекомендаций юзерами из разных тестовых групп
views.info()


# In[10]:


views.user_id.nunique()


# In[57]:


# группы примерно равны
views.exp_group.value_counts(normalize = True)


# Проверим систему сплитования на всякий случай (нет ли юзеров, попавших в обе группы), и если вдруг есть:  
#     - удалим их из обеих групп в случае незначительного количества,  
#     - пойдем смотреть систему сплитования в случае большого числа задвоенных пользователей

# In[58]:


# ищем повторных
duplicates_users = ((views[['user_id', 'exp_group']]   .drop_duplicates())  .groupby('user_id', as_index = False)  .agg({'exp_group' : 'count'})  .sort_values(by = 'exp_group', ascending = False)  .query('exp_group > 1').user_id).tolist()
duplicates_users


# In[59]:


# удаляем найденных юзеров из обеих групп
def cut_duplicates(df, lst):
    for i in range(len(lst)):
        df = df.query('user_id != @lst[@i]')
    return df

likes_cut = cut_duplicates(likes, duplicates_users)
views_cut = cut_duplicates(views, duplicates_users)


# In[60]:


views_cut.groupby('user_id').first().exp_group.value_counts()


# Как будто группы примерно равны, но проверим дополнительно биномиальным тестом

# In[10]:


# нет оснований отклонить гипотезу об отсутствии различий между группами
binomtest(32659, 65009, p=0.5)


# Группы теста и контроля почти равны по числу юзеров, но различается ли поведение пользователей в группах?  
# Сравним тест и контроль по тому, как они оценивали рекомендованные посты (*число лайков в группе / hitrate*)

# In[61]:


# табличка с группами юзеров
users_group = views_cut[['user_id', 'exp_group']].drop_duplicates()

# примерджим юзерам и группам их лайки
users_group_like = users_group.merge(likes_for_users, how = 'left').fillna(0)

# добавим колонку с фактом лайка показанной рекомендации для удобства
users_group_like['is_like'] = np.where(users_group_like.likes_count != 0, 1, 0)


# In[62]:


# посмотрим на средние метрики по группам
users_group_like.groupby('exp_group', as_index = False)[['likes_count', 'is_like']].mean()


# In[63]:


# число лайков в группе проверим критерием Манна-Уитни, поскольку лайки распределены логнормально
# видно, что различия стат значимы при уровне значимости 0,05
mannwhitneyu(
    users_group_like[users_group_like.exp_group == 'test'].likes_count,
    users_group_like[users_group_like.exp_group == 'control'].likes_count)


# ### Высчитаем общий hitrate (доля рекомендаций, в которые пользователи кликнули)  
# Основная идея - распилим список рекомендаций по рекомендованным постам, примерджим к полученным данным лайки по айдишникам юзера и поста. Так мы увидим, было ли просмотрено что - то из рекомендованного каждым юзером и время просмотра. Ну и исходя из этого оценим метрику

# In[64]:


views_cut['post_id'] = views_cut.recommendations.apply(lambda x: x[1:-1].split(' '))
views_cut = views_cut.explode('post_id')
views_cut = views_cut[views_cut.post_id != '']
views_cut.post_id = views_cut.post_id.map(int)
full_likes_views = views_cut.merge(likes_cut, how = 'left', on = ['user_id', 'post_id'])


# In[65]:


# итоговая таблица с активностями сильно выросла
full_likes_views.shape


# In[66]:


# общее число просмотров рекомендаций
full_likes_views.recommendations.nunique()


# In[67]:


# таблица с активностями, у которых зафиксировано время рекомендации и лайка соответствующих постов
with_likes = full_likes_views[full_likes_views.timestamp_y.notna()]


# In[68]:


# почти 231 тысячу ранее рекомендованных постов лайкнули юзеры
with_likes.shape


# In[144]:


# а это все показы рекомендаций, которые имеют лайки
with_likes.recommendations.nunique()


# In[141]:


# доля показов рекомендаций, где был хоть один лайк
# hitrate
138066/193268


# ### Отличаются ли метрики тестовых групп между собой и значимо ли это различие?  
# Применим бакетный подход на 100 бакетов, чтобы посчитать групповой hitrate (доля hitrate по группе/бакету). Уровень значимости останется тем же на уровне 0.05.

# In[217]:


full_likes_views_bucket = full_likes_views.copy()
full_likes_views_bucket.head()


# In[218]:


# добавим лайки юзеров
full_likes_views_bucket['likes'] = np.where(full_likes_views_bucket.timestamp_y == full_likes_views_bucket.timestamp_y, 1,0)
full_likes_views_bucket.drop(columns = 'timestamp_y',axis = 1, inplace = True)
full_likes_views_bucket.head()


# In[219]:


# подготовим бакеты
import hashlib

full_likes_views_bucket['bucket'] = full_likes_views_bucket['user_id'].apply(
    lambda x: int(hashlib.md5((str(x) + 'bbb').encode()).hexdigest(), 16) % 100
)


# In[220]:


# посчитаем лайки и просмотры по каждой рекомендации
buckets = full_likes_views_bucket.groupby(['exp_group', 'bucket', 'recommendations'], as_index = False).sum('likes')
buckets.likes = np.where(buckets.likes == 0, 0, 1)
buckets['view'] = 1
buckets.head()


# In[221]:


# сгруппируем по бакетам тестовых групп и посчитаем метрики для каждого бакета
buckets_res = buckets.groupby(['exp_group', 'bucket'], as_index = False).sum(['likes', 'views'])
buckets_res['bucket_hitrate'] = buckets_res.likes / buckets_res.view
buckets_res.groupby('exp_group').bucket_hitrate.mean()


# Видно, что в тесте метрика увеличилась на 2пп, необходимо проверить значимость изменений (используем все так же Манна - Уитни)

# In[224]:


# рассмотренные распределения различаются значимо
# hitrate в тестовой группе вырос
mannwhitneyu(buckets_res[buckets_res.exp_group == 'test'].bucket_hitrate, buckets_res[buckets_res.exp_group == 'control'].bucket_hitrate)

