library(tidytext) #Package for text mining
library(tidyverse) 
library(SnowballC) 
library(tidyquant) 
library(wordcloud) #to create word clouds
library(ggplot2)
library(plotly) 
library(stopwords) #has stop words in 50 different languages

#Setting the working directory
setwd("C:/Users/Rushad Khan/Desktop/AT3 MLAA")
#Inputting the data
data <- read_tsv('Restaurant_Reviews.tsv',quote = '')


#First Analysis
#sentiment analysis
#Looking the first 6 rows of the dataset in table format
head(data) %>% 
  knitr::kable()


#extracting sentiments from the dataset
sentimentanalysis <- data %>% 
  
  #adding the customer ID Row to the dataset
  mutate(customer_id = row_number()) %>% 
  unnest_tokens(output = word, input = Review) %>% 
  
  #Rearranging the columns in the dataset
  select(customer_id,everything()) %>% 
  
  #using a combo of get_stopwords & anti_join function to remove all the stop words
  anti_join(get_stopwords(), by = 'word') %>% 
  
  #converting each words to the stem in english language
  mutate(word = wordStem(word, language = 'english')) %>% 
  
  #Extracting positive and negative sentiments from each word
  inner_join(get_sentiments('bing'), by  = 'word')


#Looking the first 10 rows of the mutated data set in table format
#after removing all the stop words and labelling the sentiments
head(sentimentanalysis, 10) %>% 
  knitr::kable()



#Looking at the Top 25 words used in the Comments
sentimentanalysis %>% 
  count(word, sort = TRUE) %>% 
  top_n(25) %>% 
  ggplot(aes(fct_reorder(word,n),n))+
  geom_col(aes(fill = word), show.legend = FALSE)+
  coord_flip()+
  labs(x = '', y = 'count')+
  theme_tq()

#Looking at Top words used in positive and negative comments
plot1 <- sentimentanalysis %>% 
  mutate(Liked = as.factor(Liked)) %>% 
  group_by(Liked) %>% 
  count(word, sort = TRUE) %>% 
  top_n(10) %>% 
  ungroup() %>% 
  ggplot(aes(fct_reorder(word,n), n,fill = Liked))+
  geom_col(show.legend = FALSE)+
  coord_flip()+
  facet_wrap(~ Liked, scales = 'free')+
  labs(title = 'Top 10 words used for Comments labelled as Liked = 1 and Disliked = 0',
       x = '', y = 'count')+
  theme_tq()+
  theme(legend.position = 'none')

ggplotly(plot1)

#positive words like, good, and best appear frequently in negative comments
#indicating that they were likely used with other words to describe the service. 
#As a result, rather than looking for single words, we may need to hunt for combinations of words

#Looking at top positive and negative words used in positive and negative comments
plot2 <- sentimentanalysis %>% 
  count(word,sentiment, sort = T) %>%
  group_by(sentiment) %>% 
  top_n(10) %>% 
  ungroup() %>% 
  ggplot(aes(x = fct_reorder(word,n), y = n, fill = word))+
  geom_col(show.legend = FALSE) +
  coord_flip()+
  facet_wrap(~ sentiment, scales = 'free_y')+
  labs(title= 'Top 10 positve and negative words used for Comments labelled as Liked = 1 and Disliked = 0', 
       y = 'count', x = '')+
  theme_tq()+
  theme(legend.position = 'none')

ggplotly(plot2)

#Counting the words sentiment wise
countword <- sentimentanalysis %>% 
  count(word,sentiment) 

#Creating a word cloud
wordcloud(words = countword$word, freq = countword$n,max.words = 200,
          colors = brewer.pal(8,'Dark2'),fixed.asp = T, random.order = F
)

#Creating a word cloud by splitting positive and negative words
sentimentanalysis %>% 
  count(word,sentiment, sort = T) %>% 
  spread(key = sentiment, value = n, fill = 0) %>% 
  column_to_rownames('word') %>% 
  as.matrix() %>% 
  comparison.cloud(max.words = 100,
                   title.colors = c('red','green'),
                   title.size = 2,
                   match.colors = F,colors = c('red','blue'))




##Second Text Analysis
##Two Consecutive word relations  

#separating the data into pair of consecutive words(Bigrams)
#Then counting the number of times that bigram was used
consecutivewords <- data %>% 
  mutate(customer_id = row_number()) %>% 
  select(customer_id, everything()) %>% 
  unnest_tokens(input = Review, output = bigram, token = 'ngrams', n = 2) %>% 
  separate(bigram, into = c('word1','word2')) %>%
  
  
  #removing all the stop words and joining the rws
  filter(!word1 %in% get_stopwords(source = 'smart')$word,
         !word2 %in% get_stopwords(source = 'smart')$word) %>% 
  unite(bigram, word1,word2, sep = ' ') %>% 
  count(Liked,bigram, sort = T)


#Looking the first 10 rows of the mutated data set in table format
#N column shows the number of times a bigram was used in the comments
head(consecutivewords,10) %>% 
  knitr::kable()

#Top 10 bigrams used for Comments labelled as Liked = 1 and Disliked = 0'
consecutivewords %>% 
  mutate(Liked = as.factor(Liked)) %>% 
  group_by(Liked) %>% 
  slice(1:10) %>% 
  ungroup() %>% 
  ggplot(aes(fct_reorder(bigram,n), n))+
  geom_col(aes(fill = Liked), show.legend = F)+
  coord_flip()+
  facet_wrap(~ Liked, scales = 'free')+
  theme_tq()+
  labs(title = 'Top 10 bigrams used for Comments labelled as Liked = 1 and Disliked = 0',
       x = '', y = 'count')

#Creating a word cloud by splitting positive and negative bigrams
consecutivewords %>% 
  mutate(Liked = ifelse(Liked == 1, 'Liked','Not Liked')) %>% 
  spread(Liked, n, fill = 0) %>% 
  column_to_rownames('bigram') %>% 
  as.matrix() %>% 
  comparison.cloud(max.words = 50,scale = c(2,0.3),
                   title.size = 1)
