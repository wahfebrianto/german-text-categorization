import pickle
import numpy as np
from preprocessing import preprocessing_text

keywords = ["stok","bay","werbesteuer","umsst","aktivbank","rechnungsausgleich","innoscripta","contabo","strato","global","j2","hetzner","online","norton","naturenergie","upwrkescrow","auftraggeber","auftrag","telekom","sipgate","toplink","1u1","telecom","telekommunikations","komm","telenova","billpay","aral","tankstelle","agip","warngau","moritz","knopp","fahrtkosten","lufthan","auslage","bravofly","airbnb","auslagen","sepaÜberweisung","knappschaftbahnsee","bosch","bkk","beitraege","lohn","gehalt","aok","bmw","betriebskrankenkasse","lohnabrechnung","auszahlungsbetrag","bonus","sepaÜberweisung","dakgesundheit","mobil","oil","beitrag","ikk","classic","krankenkasse","lohnst","reifen","widholzer","bundeskasse","weiden","reifendienste","pneuhage","hdi","kraftfahrtversicherung","autoservice","böttcher","otto","schaefersho","furnishyourspace","ikea","depot","alternate","weberbuero","lizengo","mindfactory","amazon","facility","sasse","management","wunderagent","miete","trockenbau","rent","voip","lebensversicherung","real","estate","onlinemarketing","content5","handbuch","experten","swm","versorgungs","operngrill","fechcon","dienstwohnung","grundbesitz","lohndata","rechtsanwalte","partner","landesjustizkasse","gkk","partners","rechtsberatung","eutop","international"]
categories = ["steuer", "aktivbank", "it", "telekom", "reisekosten", "personalkosten", "dienstwagen", "invest", "miete", "dienstleister", "miete_it", "steuerberater", "recht", "joos", "sonstiges"]
connections = [
  { "category": "steuer", "keywords": ["stok", "bay", "werbesteuer", "umsst"] },
  {
    "category": "aktivbank",
    "keywords": ["aktivbank", "rechnungsausgleich", "innoscripta"]
  },
  {
    "category": "it",
    "keywords": [
      "contabo",
      "strato",
      "global",
      "j2",
      "hetzner",
      "online",
      "norton",
      "naturenergie",
      "upwrkescrow",
      "auftraggeber",
      "auftrag"
    ]
  },
  {
    "category": "telekom",
    "keywords": [
      "telekom",
      "sipgate",
      "toplink",
      "1u1",
      "telecom",
      "telekommunikations",
      "komm",
      "telenova"
    ]
  },
  {
    "category": "reisekosten",
    "keywords": [
      "billpay",
      "aral",
      "tankstelle",
      "agip",
      "warngau",
      "moritz",
      "knopp",
      "fahrtkosten",
      "lufthan",
      "auslage",
      "bravofly",
      "airbnb",
      "auslagen",
      "sepaÜberweisung"
    ]
  },
  {
    "category": "personalkosten",
    "keywords": [
      "knappschaftbahnsee",
      "bosch",
      "bkk",
      "beitraege",
      "lohn",
      "gehalt",
      "aok",
      "bmw",
      "bkk",
      "betriebskrankenkasse",
      "lohnabrechnung",
      "auszahlungsbetrag",
      "bonus",
      "sepaÜberweisung",
      "innoscripta",
      "dakgesundheit",
      "bmw",
      "mobil",
      "oil",
      "beitrag",
      "ikk",
      "classic",
      "krankenkasse",
      "lohnst"
    ]
  },
  {
    "category": "dienstwagen",
    "keywords": [
      "reifen",
      "widholzer",
      "bundeskasse",
      "weiden",
      "reifendienste",
      "pneuhage",
      "bmw",
      "innoscripta",
      "hdi",
      "global",
      "kraftfahrtversicherung",
      "autoservice"
    ]
  },
  {
    "category": "invest",
    "keywords": [
      "böttcher",
      "otto",
      "schaefersho",
      "furnishyourspace",
      "ikea",
      "depot",
      "alternate",
      "weberbuero",
      "lizengo",
      "mindfactory",
      "amazon"
    ]
  },
  {
    "category": "miete",
    "keywords": [
      "facility",
      "sasse",
      "management",
      "wunderagent",
      "miete",
      "trockenbau",
      "rent",
      "voip",
      "lebensversicherung",
      "real",
      "estate"
    ]
  },
  {
    "category": "dienstleister",
    "keywords": [
      "auftrag",
      "onlinemarketing",
      "innoscripta",
      "content5",
      "handbuch",
      "experten"
    ]
  },
  {
    "category": "miete_it",
    "keywords": [
      "swm",
      "versorgungs",
      "operngrill",
      "fechcon",
      "miete",
      "dienstwohnung",
      "grundbesitz"
    ]
  },
  {
    "category": "steuerberater",
    "keywords": ["lohndata", "rechtsanwalte", "partner"]
  },
  {
    "category": "recht",
    "keywords": [
      "landesjustizkasse",
      "gkk",
      "partners",
      "rechtsanwalte",
      "rechtsberatung"
    ]
  },
  { "category": "joos", "keywords": ["eutop", "international"] },
  { "category": "sonstiges", "keywords": [] }
]

adjacency_matrix = np.zeros([len(keywords), len(categories)], np.float32)

def init_adjecency_matrix():
    for keyword_index in range(len(keywords)):
        keyword = keywords[keyword_index]
        count = 0
        for connection_index in range(len(connections)):
            connection = connections[connection_index]
            if keyword in connection['keywords']:
                adjacency_matrix[keyword_index][connection_index] = 1
                count = count + 1
            else:
                adjacency_matrix[keyword_index][connection_index] = 0
        if count > 0:
            adjacency_matrix[keyword_index] = adjacency_matrix[keyword_index] / count
        else:
            print(keyword)

with open('data/123456789/categories.dictionary', 'wb') as categories_dictionary:
    pickle.dump(categories, categories_dictionary)

init_adjecency_matrix()
with open('data/123456789/adjecency_matrix.dictionary', 'wb') as adjecency_matrix_dictionary:
    pickle.dump(adjacency_matrix, adjecency_matrix_dictionary)

with open('data/123456789/keywords.dictionary', 'wb') as keywords_dictionary:
    pickle.dump(preprocessing_text(keywords), keywords_dictionary)
