(ns mnist.core
 (:require [jutsu.ai.core :as ai]) 
 (:import [org.deeplearning4j.nn.conf.preprocessor CnnToFeedForwardPreProcessor ]))


(def mnist-net-config
	[:seed 123
   :activation :relu
   :updater :nesterovs
   :iterations 1
   :optimization-algo :sgd
   :learning-rate 0.006
   :regularization true
   :l2 1e-4
  :layers 
		[[:dense [:n-in 784 :n-out 100 :activation :relu]]
	 	 [:output :negative-log-likelihood [:n-in 100 :n-out 10 :activation :softmax]]]
    :input-pre-processors {0 (CnnToFeedForwardPreProcessor. 28 28 1)}])

(def mnist-net
	(-> mnist-net-config
		  ai/network
		  ai/initialize-net))

(defn mnist-train []
  (let [mnist-training-iterator (ai/classification-dir-labeled-image-iterator "data/mnist/training" 28 28 1 128 10 123)
        mnist-testing-iterator  (ai/classification-dir-labeled-image-iterator "data/mnist/test" 28 28 1 128 10 123)]
     (-> mnist-net
         (ai/train-net! 1 mnist-training-iterator)
         (ai/save-model "mnist-model.nn"))
     (println (ai/evaluate mnist-net mnist-testing-iterator))))

(mnist-train)
