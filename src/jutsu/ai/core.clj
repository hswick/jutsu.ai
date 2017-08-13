(ns jutsu.ai.core           
  (:import [org.datavec.api.split FileSplit]
           [org.datavec.api.util ClassPathResource]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator]
           [org.datavec.api.records.reader.impl.csv CSVRecordReader]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.conf Updater]
           [org.deeplearning4j.nn.conf.layers DenseLayer$Builder]
           [org.nd4j.linalg.activations Activation]
           [org.deeplearning4j.nn.conf.layers OutputLayer$Builder]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.util ModelSerializer]))

(defn regression-csv-iterator [filename batch-size label-index]
  (let [path (-> (ClassPathResource. filename)
              (.getFile))
        rr (CSVRecordReader.)]
    (.initialize rr (FileSplit. path))
    (RecordReaderDataSetIterator. rr nil batch-size label-index 1 true)))

(defn classification-csv-iterator [filename batch-size label-index num-possible-labels]
  (let [path (-> (ClassPathResource. filename)
                 (.getFile))
        rr (CSVRecordReader.)]
    (.initialize rr (FileSplit. path))
    (RecordReaderDataSetIterator. rr batch-size label-index num-possible-labels)))

(defn default-regression-options []
  {:seed 12345
   :iterations 1
   :optimization-algo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :learning-rate 0.01
   :weight-init (WeightInit/XAVIER)
   :updater (Updater/NESTEROVS)
   :momentum 0.9
   :activation (Activation/TANH)
   :pretrain false
   :backprop true
   :num-hidden-nodes 50
   :loss-function (LossFunctions$LossFunction/MSE)})

(defn regression-net
  ([num-in num-out] (regression-net num-in num-out {}))
  ([num-in num-out options-map]
   (let [final-map (merge (default-regression-options) options-map)]
    (-> (NeuralNetConfiguration$Builder.)
      (.seed (:seed final-map))
      (.iterations (:iterations final-map))
      (.optimizationAlgo (:optimization-algo final-map))
      (.learningRate (:learning-rate final-map))
      (.weightInit (:weight-init final-map))
      (.updater (:updater final-map))
      (.momentum (:momentum final-map))
      (.list)
      (.layer 0 (-> (DenseLayer$Builder.)
                    (.nIn num-in)
                    (.nOut (:num-hidden-nodes final-map))
                    (.activation (:activation final-map))
                    (.build)))
      (.layer 1 (-> (DenseLayer$Builder.)
                    (.nIn (:num-hidden-nodes final-map))
                    (.nOut (:num-hidden-nodes final-map))
                    (.activation (:activation final-map))
                    (.build)))
      (.layer 2 (-> (OutputLayer$Builder. (:loss-function final-map))
                    (.activation (Activation/IDENTITY))
                    (.nIn (:num-hidden-nodes final-map))
                    (.nOut num-out)
                    (.build)))
      (.pretrain (:pretrain final-map))
      (.backprop (:backprop final-map))
      (.build)
      (MultiLayerNetwork.)))))

(defn default-classification-options []
  {:seed 12345
   :optimization-algo (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :iterations 1
   :learning-rate 0.006
   :updater (Updater/NESTEROVS)
   :momentum 0.9
   :regularization true
   :l2 1e-4
   :weight-init (WeightInit/XAVIER)
   :activation (Activation/RELU)
   :loss-function (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
   :pretrain false
   :backprop true})

(defn classification-net
  ([num-in num-out] (classification-net num-in num-out {}))
  ([num-in num-out options-map]
   (let [final-map (merge (default-classification-options) options-map)]
     (-> (NeuralNetConfiguration$Builder.)
       (.seed (:seed final-map))
       (.optimizationAlgo (:optimization-algo final-map))
       (.iterations (:iterations final-map))
       (.learningRate (:learning-rate final-map))
       (.updater (:updater final-map))
       (.momentum (:momentum final-map))
       (.regularization (:regularization final-map))
       (.l2 (:l2 final-map))
       (.list)
       (.layer 0 (-> (DenseLayer$Builder.)
                     (.nIn num-in)
                     (.nOut (:num-hidden-nodes final-map))
                     (.activation (:activation final-map))
                     (.weightInit (:weight-init final-map))
                     (.build)))
       (.layer 1 (-> (OutputLayer$Builder. (:loss-function final-map))
                     (.nIn (:num-hidden-nodes final-map))
                     (.nOut num-out)
                     (.activation (Activation/SOFTMAX))
                     (.weightInit (:weight-init final-map))
                     (.build)))
       (.pretrain (:pretrain final-map))
       (.backprop (:backprop final-map))
       (.build)
       (MultiLayerNetwork.)))))

;;set true for online learning
(defn save-model 
  ([net filename]
   (save-model net filename false))
  ([net filename ready-for-more]
   (ModelSerializer/writeModel net (java.io.File. filename) ready-for-more)))

(defn load-model [filename]
   (ModelSerializer/restoreMultiLayerNetwork filename))

(defn initialize-net!
  ([net] (initialize-net! net (list (ScoreIterationListener. 1))))
  ([net listeners]
   (.init net)
   (.setListeners net listeners)
   net))

(defn train-net! [net epochs dataset-iterator]
  (doseq [n (range 0 epochs)]
    (.reset dataset-iterator)
    (.fit net dataset-iterator))
  net)

(defn parse-shorthand [topology]
  (map (fn [layer]
         {:in (nth layer 0)
          :out (nth layer 1)
          :activation (nth layer 2)})
    topology))

;;throw error if now activation key
(defn interpret-activation [layer])

;;pass in shorthand with vectors
(defn parse-topology [topology]
  (let [topo (if (= clojure.lang.PersistentVector (class (first topology)))
               (parse-shorthand topology)
               topology)
        proper-topo (map interpret-activation topo)]))
