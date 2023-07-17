# NLP

NLP (Natural Language Processing) enables computers to understand and generate human language. It has applications in speech recognition, machine translation, sentiment analysis, and text classification.


# Key steps in NLP:

<ol>
  <li>Text pre-processing: Cleaning and transforming raw text data for analysis.</li>

  <li>Tokenization: Breaking text into individual words or tokens.</li>

  <li>Stemming: Reducing words to their base form.</li>

  <li>Stopword removal: Removing common, uninformative words.</li>

  <li>Part-of-speech tagging: Assigning a grammatical tag to each word.</li>
</ol>

# NLP tasks:
<ol>
  <li>Sentiment analysis: Identifying sentiment in the text (positive, negative, neutral).</li>

  <li>Text classification: Assigning categories or labels to text.</li>

  <li>Machine learning models: Support vector machines, logistic regression, and neural networks.</li>
</ol>

# Benefits:
<ol>
  <li>Social media monitoring</li>

  <li>Customer feedback analysis</li>

  <li>Brand reputation management</li>

  <li>Spam filtering</li>
</ol>

<h1>Text Preprocessing - NLP</h1>

<lu>

 <h3> <strong> BOW: </strong> </h3>The Bag of Words (BoW) is a popular technique in natural language processing (NLP) for text representation. It disregards the sequence and context of words and focuses on the frequency of individual words in a document. Here's a breakdown of the BoW process:
<lu>
  <li>Tokenization: The text is divided into individual words or tokens. </li>
  <li>Vocabulary creation: A unique set of words, called vocabulary, is created from the tokenized words across all documents.</li>
  <li>Document representation: Each document is represented as a numerical vector based on the frequency of words in the vocabulary.</li>
  <li>Here's an example of how to implement the Bag of Words technique in Python using the sci-kit-learn library:</li>
</lu>

<pre>  
<code>
  from sklearn.feature_extraction.text import CountVectorizer
  corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(corpus)
  print("Vocabulary:")
  print(vectorizer.get_feature_names())
  print("Document-term matrix:")
  print(X.toarray())
</code>
  Output: <br>
<code>
  Vocabulary:
  ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
  Document-term matrix:
  [[0 1 1 1 0 0 1 0 1]
   [0 2 0 1 0 1 1 0 1]
   [1 0 0 1 1 0 1 1 1]
   [0 1 1 1 0 0 1 0 1]]
</code>
</pre>
  <br>
<h3><strong>TF-IDF:</strong></h3> TF-IDF stands for Term Frequency-Inverse Document Frequency and is another commonly used technique for text representation in NLP. It calculates a weight for each word in a document based on its frequency in the document and its rarity across all documents in the corpus. Here's an example of how to implement TF-IDF in Python using the scikit-learn library:
<pre>
<code> <br>
  from sklearn.feature_extraction.text import TfidfVectorizer
  corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
  ]  
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(corpus)
  print("Vocabulary:")
  print(vectorizer.get_feature_names())
  print("TF-IDF matrix:")
  print(X.toarray())
</code> <br>

Output: <br>
<code>
  Vocabulary:
  ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
  TF-IDF matrix:
  [[0.         0.44451454 0.6316672  0.44451454 0.         0.
    0.44451454 0.         0.44451454]
   [0.         0.88822908 0.         0.31544431 0.         0.47212057
    0.31544431 0.         0.31544431]
   [0.52640543 0.         0.         0.3695515  0.52640543 0.
    0.3695515  0.52640543 0.3695515 ]
   [0.         0.44451454 0.6316672  0.44451454 0.         0.
    0.44451454 0.         0.44451454]]

</code> <br>
</pre>
<h3><strong>Word2Vec - AvgWord2Vec:</strong></h3><br>
  <strong> A-) </strong> Word2Vec:
    Word2Vec is a word embedding technique in natural language processing (NLP) that represents words as dense vectors  
    in a continuous vector space. <br> It captures semantic and syntactic relationships between words by learning from large 
    text corpora. <br> The main idea behind Word2Vec is that words with similar meanings tend to appear in similar contexts. It 
    can be trained using two main approaches: Continuous Bag of Words (CBOW) and Skip-gram. <br> <br>
    CBOW predicts the target word based on the surrounding context words. It takes the average of the context word vectors to predict the target word vector. Skip-gram, on the other hand, predicts the context words given the target word. It uses the target word vector to predict the context word vectors. Both approaches learn to generate meaningful word representations that encode semantic relationships.

The resulting word vectors from Word2Vec capture semantic similarities between words. Words with similar meanings are closer together in the vector space, allowing for operations like word analogies and semantic comparisons.

  <strong> B-) </strong> Average Word2Vec:
    Average Word2Vec is a technique that represents a sentence or document by averaging the word vectors of its constituent words, which are obtained using a pre-trained Word2Vec model.  <br>
    It aims to capture the overall semantic meaning of the sentence or document by considering the meanings of its individual words.
<br>
    Steps:
<lu>
  <li>Tokenization:  The sentence or document is divided into individual words or tokens. </li>
  <li>Word vector retrieval: The pre-trained Word2Vec model is used to obtain the word vectors for each word in the sentence or document </li>
  <li>Averaging: The word vectors are averaged element-wise to create a single vector representation.</li>
  <li>By averaging the word vectors, Average Word2Vec provides a fixed-length vector representation for variable-length sentences or documents. This representation can be used as input for various machine learning algorithms or similarity calculations.</li>
</lu>
<br> It's worth noting that when using Average Word2Vec, words not present in the pre-trained Word2Vec model will be omitted, and stopwords or very rare words may have less impact on the resulting representation due to averaging.

Word2Vec and Average Word2Vec are widely used techniques in NLP for capturing word semantics and representing sentences or documents in a continuous vector space, enabling various downstream tasks such as document classification, sentiment analysis, and information retrieval.
<br>
<h3><strong>Artifical Neural Network</strong></h3>
<lu>
  
  <li>Purpose: ANNs are computational models inspired by the structure and functioning of biological neural networks. They consist of interconnected artificial neurons organized in layers. ANNs are designed to learn from data and perform tasks such as classification, regression, pattern recognition, and more.</li>
  <li>Input and Output: ANNs take input data, which can be numerical, categorical, or even image pixels, and pass it through the network layers to produce an output. The input data is transformed through weighted connections and activation functions within the network.</li>
  <li>Structure: ANNs are composed of an input layer, one or more hidden layers, and an output layer. Each layer consists of multiple artificial neurons (nodes). The connections between neurons have associated weights that are adjusted during the training process to optimize the network's performance.</li>
  <li>Training: ANNs are trained using various learning algorithms such as backpropagation. During training, the network iteratively adjusts the weights to minimize the difference between the predicted output and the desired output. This process involves forward propagation (calculating outputs) and backward propagation (updating weights based on the error).</li>
  <li>Flexibility: ANNs can handle complex patterns and non-linear relationships in data, making them capable of solving a wide range of problems. They can learn from large datasets and generalize well to unseen data when properly trained.
</lu>
</lu>
<br>
<div style="text-align: center;">
  <img src="https://miro.medium.com/v2/resize:fit:600/format:webp/1*_oQyazP96Ki7SNMkR1msQg.png" alt="Image Description">
</div>

