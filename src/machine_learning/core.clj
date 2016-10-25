(ns machine-learning.core
  (:import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
           org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
           org.deeplearning4j.eval.Evaluation
           org.deeplearning4j.nn.api.OptimizationAlgorithm
           org.deeplearning4j.nn.conf.MultiLayerConfiguration$Builder
           org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
           org.deeplearning4j.nn.conf.Updater
           org.deeplearning4j.nn.conf.layers.DenseLayer$Builder
           org.deeplearning4j.nn.conf.layers.OutputLayer$Builder
           org.deeplearning4j.nn.multilayer.MultiLayerNetwork
           org.deeplearning4j.nn.weights.WeightInit
           org.deeplearning4j.optimize.listeners.ScoreIterationListener
           org.nd4j.linalg.api.ndarray.INDArray
           org.nd4j.linalg.dataset.DataSet
           org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction
           org.slf4j.Logger
           org.slf4j.LoggerFactory))

(defn build-conf [rows cols outputs rng-seed]
  (.. (NeuralNetConfiguration$Builder.)
      (seed rng-seed)
      (optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
      (iterations 1)
      (learningRate 0.006)
      (updater Updater/NESTEROVS)
      (momentum 0.9)
      (regularization true)
      (l2 1e-4)
      (list)
      (layer 0 (.. (DenseLayer$Builder.)
                   (nIn (* rows cols))
                   (nOut 1000)
                   (activation "relu")
                   (weightInit WeightInit/XAVIER)
                   (build)))
      (layer 1 (.. (OutputLayer$Builder. LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)
                   (nIn 1000)
                   (nOut outputs)
                   (activation "softmax")
                   (weightInit WeightInit/XAVIER)
                   (build)))
      (pretrain false)
      (backprop true)
      (build)))

(defn run []
  (let [rows 28
        cols 28
        outputs 10
        batch-size 128
        rng-seed 123
        num-epochs 15

        mnist-train (MnistDataSetIterator. batch-size true rng-seed)
        mnist-test (MnistDataSetIterator. batch-size false rng-seed)
        _ (println "Building model....")
        conf (build-conf rows cols outputs rng-seed)
        model (MultiLayerNetwork. conf)]

    (.init model)
    (.setListeners model [(ScoreIterationListener. 1)])

    (println "Training model.....")

    (dotimes [i num-epochs]
      (.fit model mnist-train))

    (println "evaluate model.....")

    (let [eval (Evaluation. outputs)]
      (doseq [next (seq mnist-test)]
        (let [output (.output model (.getFeatureMatrix next))]
          (.eval eval (.getLabels next) output)))


      (println (.stats eval))))

  (println "******DONE******"))
