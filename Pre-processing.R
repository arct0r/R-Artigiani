library(readxl)
library(quanteda)
library(quanteda.textstats)

setwd("C:/Users/WilliamSanteramo/OneDrive - ITS Angelo Rizzoli/Documenti/UFS/07 programmazione R/PROGETTO")

StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")

Ita_StoresReview <- StoresReview[StoresReview$lang_value == "it" | is.na(StoresReview$lang_value) == TRUE,]

Testo_Corpus <- corpus(na.omit(Ita_StoresReview$text))
textstat_summary(Testo_Corpus)


Testo_dfm <- dfm(tokens(Testo_Corpus,
                              remove_numbers = TRUE,
                              remove_punct = TRUE,
                              remove_symbols = TRUE) %>%
  tokens_tolower() %>% 
  tokens_remove(c(stopwords("italian"))) %>%
  tokens_wordstem(language = "italian"))

topfeatures(Testo_dfm,100)
topfeatures(Contenitore,100)

summary(Testo_dfm)

Contenitore <- dfm_trim(Testo_dfm,
                        min_termfreq = 10,
                        min_docfreq = 2)
summary(Contenitore)

# allenamento dell'algoritmo.
# estrazione campionaria randomica
# 200 testi a testa
# codebook, documento word per noi
# driver serve il dizionario (normale o newsmap).