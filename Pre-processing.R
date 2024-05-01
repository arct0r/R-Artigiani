# librerie e dataset ----
library(readxl)
library(quanteda)
library(quanteda.textstats)
library(ggplot2)

# setwd("C:/Users/WilliamSanteramo/OneDrive - ITS Angelo Rizzoli/Documenti/UFS/07 programmazione R/PROGETTO")
# Vecchio setwd manuale

library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))
# Questo codice magico permette di settare automaticamente la working directory nella cartella in cui si trova lo script
# è utile perchè, lavorando con git, cloniamo continuamente e non si può stare a cambiare ogni volta il path per ogni pc diverso

StoresReview <- read_excel("GRUPPO 3-4-5. Industry elettronica.xlsx")

Ita_StoresReview <- StoresReview[StoresReview$lang_value == "it" | is.na(StoresReview$lang_value) == TRUE,]


# PRE- PROCESSING DFM ----
Testo_Corpus <- corpus(na.omit(Ita_StoresReview$text))
textstat_summary(Testo_Corpus)


Testo_dfm <- dfm(tokens(Testo_Corpus,
                              remove_numbers = TRUE,
                              remove_punct = TRUE,
                              remove_symbols = TRUE) %>%
  tokens_tolower() %>% 
  tokens_remove(c(stopwords("italian"))) %>%
  tokens_wordstem(language = "italian"))

topfeatures(Testo_finito,100)

summary(Testo_dfm)

Testo_finito <- dfm_trim(Testo_dfm,
                        min_termfreq = 10,
                        #max_termfreq = 500,
                        min_docfreq = 2)

# ANALISI ----
Tabella_descrittiva <- textstat_frequency(Testo_finito, n =500)

set.seed(000)
Review_training <- sample(Testo_Corpus, size = 200, replace = FALSE)

Review_test <- Testo_Corpus[!(Testo_Corpus %in% Review_training)]

setequal(Testo_Corpus, union(Review_test, Review_training))

Lavoro <- list(
  William = rep("", 50),
  Davide = rep("", 50),
  Maddalena = rep("", 50),
  Giacomo = rep("", 50)
)
k <- 0

for (i in 1:4){
  Lavoro[[i]] <- Review_training[(k+1) : (50 * i)]
  k <- 50 * i
}

print(Lavoro)


# SUGGERIMENTI ----
# allenamento dell'algoritmo.
# estrazione campionaria randomica
# 200 testi a testa
# codebook, documento word per noi
# driver serve il dizionario (normale o newsmap).