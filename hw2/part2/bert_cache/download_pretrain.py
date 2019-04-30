from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForSequenceClassification

BertForSequenceClassification.from_pretrained('bert-large-uncased', cache_dir='./', num_labels=5)
BertTokenizer.from_pretrained('bert-large-uncased', cache_dir='./')

BertForSequenceClassification.from_pretrained('bert-large-cased', cache_dir='./', num_labels=5)
BertTokenizer.from_pretrained('bert-large-cased', cache_dir='./')

BertModel.from_pretrained('bert-large-cased', cache_dir='./')
BertTokenizer.from_pretrained('bert-large-cased', cache_dir='./')

BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./')
BertModel.from_pretrained('bert-base-uncased', cache_dir='./')

