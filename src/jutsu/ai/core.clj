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
           [org.deeplearning4j.util ModelSerializer]
           [org.deeplearning4j.nn.conf.layers RBM$Builder]))

(defn regression-csv-iterator [filename batch-size label-index]
  (let [path (-> (ClassPathResource. filename)
              (.getFile))
        rr (CSVRecordReader.)]
    (.initialize rr (FileSplit. path))
    (RecordReaderDataSetIterator. rr nil batch-size label-index -1 true)))

(defn classification-csv-iterator [filename batch-size label-index num-possible-labels]
  (let [path (-> (ClassPathResource. filename)
                 (.getFile))
        rr (CSVRecordReader.)]
    (.initialize rr (FileSplit. path))
    (RecordReaderDataSetIterator. rr batch-size label-index num-possible-labels)))

(def activation-options
  {:relu (Activation/RELU)
   :identity (Activation/IDENTITY)
   :softmax (Activation/SOFTMAX)
   :tanh (Activation/TANH)
   :sigmoid (Activation/SIGMOID)})

(def layer-builders
  {:default (fn [] (DenseLayer$Builder.))
   :rbm (fn [] (RBM$Builder.))})

(def loss-functions
  {:mse (LossFunctions$LossFunction/MSE)
   :negative-log-likelihood (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
   :kl-divergence (LossFunctions$LossFunction/KL_DIVERGENCE)})

(def optimization-algos
  {:sgd (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :stochastic-gradient-descent (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
   :line-gradient-descent (OptimizationAlgorithm/LINE_GRADIENT_DESCENT)})

(defn default-regression-options []
  {:seed 12345
   :iterations 1
   :optimization-algo :sgd
   :learning-rate 0.01
   :weight-init (WeightInit/XAVIER)
   :updater (Updater/NESTEROVS)
   :momentum 0.9
   :activation (Activation/RELU)
   :pretrain false
   :backprop true
   :num-hidden-nodes 50
   :output-loss-function :mse
   :layer-builder :default})

(defn default-classification-options []
  {:seed 12345
   :optimization-algo :sgd
   :iterations 1
   :learning-rate 0.006
   :updater (Updater/NESTEROVS)
   :momentum 0.9
   :regularization true
   :l2 1e-4
   :weight-init (WeightInit/XAVIER)
   :output-loss-function :negative-log-likelihood
   :pretrain false
   :backprop true
   :layer-builder :default})

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

;;Should add an argument for loss function
(defn parse-shorthand [topology]
  (map (fn [layer]
         {:in (nth layer 0)
          :out (nth layer 1)
          :activation (nth layer 2)
          :loss (get layer 3)});;using get instead of nth makes it optional
    topology))

;;throw error if now activation key
(defn interpret-layer [i layer topo-count options-map]
    (fn [net]
      (-> net
        (.layer i 
          (-> (if (= i topo-count)
                  (OutputLayer$Builder. (get loss-functions 
                                          (if (nil? (:loss layer))
                                            :mse
                                            (:loss layer))))
                  ((get layer-builders (:layer-builder options-map))))
              (.nIn (:in layer))
              (.nOut (:out layer))
              ((fn [layer-config]
                (if (and (:loss layer) (not= i topo-count))
                 (.lossFunction layer-config (get loss-functions (:loss layer))))
                layer-config))
              ((fn [layer-config]
                (when-let [activation (:activation layer)]
                 (.activation layer-config 
                  (get activation-options (:activation layer))))
                layer-config));should emit error when cant find function
              (.build))))))

;;pass in shorthand with vectors
;;Returns a vector of functions to be called on the net
(defn parse-topology [topology options-map]
  (let [topo (if (= clojure.lang.PersistentVector (class (first topology)))
               (parse-shorthand topology)
               topology)
        topo-count (count topo)
        proper-topo (doall (map-indexed (fn [i layer]
                                          (interpret-layer i layer (dec topo-count) options-map)) topo))]
    proper-topo))

(defn initialize-layers [net parsed-topology]
  ((apply comp parsed-topology) net))

(defn network [topology options-map]
  (let [parsed-topology (parse-topology topology options-map)]
    (-> (NeuralNetConfiguration$Builder.)
      (.seed (:seed options-map))
      (.iterations (:iterations options-map))
      (.optimizationAlgo (get optimization-algos (:optimization-algo options-map)))
      (.learningRate (:learning-rate options-map))
      (.weightInit (:weight-init options-map))
      (.updater (:updater options-map))
      (.momentum (:momentum options-map))
      (.list)
      (initialize-layers parsed-topology)
      (.pretrain (:pretrain options-map))
      (.backprop (:backprop options-map))
      (.build)
      (MultiLayerNetwork.))))
