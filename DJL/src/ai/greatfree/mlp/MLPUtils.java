package ai.greatfree.mlp;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.RandomUtils;
import ai.greatfree.linearregression.DataPoints;
import ai.greatfree.linearregression.SyntheticData;
import ai.greatfree.linearregression.LNTraining;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;

/**
 * Created: 08/17/2024, Bing Li
 */
final class MLPUtils
{
	// utility functions for shuffle data
	public static void swap(NDArray arr, int i, int j)
	{
		float tmp = arr.getFloat(i);
		arr.set(new NDIndex(i), arr.getFloat(j));
		arr.set(new NDIndex(j), tmp);
	}

	public static NDArray shuffle(NDArray arr)
	{
		int size = (int) arr.size();
		Random rnd = RandomUtils.RANDOM;
		for (int i = Math.toIntExact(size) - 1; i > 0; --i)
		{
			swap(arr, i, rnd.nextInt(i));
		}
		return arr;
	}

	public static NDArray l2Penalty(NDArray w)
	{
		return ((w.pow(2)).sum()).div(2);
	}

	public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle)
	{
		return new ArrayDataset.Builder().setData(features) // set the features
				.optLabels(labels) // set the labels
				.setSampling(batchSize, shuffle) // set the batch size and random sampling
				.build();
	}

	public static void train(String fitting, NDArray trainFeatures, NDArray testFeatures, NDArray trainLabels, NDArray testLabels, int nDegree, int logInterval, int numEpochs, double[] trainLoss, double[] testLoss, double[] epochCount) throws IOException, TranslateException
	{
		Loss l2Loss = Loss.l2Loss();
		try (NDManager manager = NDManager.newBaseManager())
		{
			Tracker lrt = Tracker.fixed(0.01f);
			Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
			DefaultTrainingConfig config = new DefaultTrainingConfig(l2Loss).optDevices(Engine.getInstance().getDevices(1)) // single
																															// GPU
					.optOptimizer(sgd) // Optimizer (loss function)
					.addTrainingListeners(TrainingListener.Defaults.basic()); // Logging

			Model model = Model.newInstance("mlp");
			SequentialBlock net = new SequentialBlock();
			// Switch off the bias since we already catered for it in the polynomial
			// features
			Linear linearBlock = Linear.builder().optBias(false).setUnits(1).build();
			net.add(linearBlock);

			model.setBlock(net);
			Trainer trainer = model.newTrainer(config);

			int batchSize = Math.min(10, (int) trainLabels.getShape().get(0));

			ArrayDataset trainIter = loadArray(trainFeatures, trainLabels, batchSize, true);
			ArrayDataset testIter = loadArray(testFeatures, testLabels, batchSize, true);

			trainer.initialize(new Shape(1, nDegree));
			System.out.println("Start Training...");
			for (int epoch = 1; epoch <= numEpochs; epoch++)
			{
				// Iterate over dataset
				for (Batch batch : trainer.iterateDataset(trainIter))
				{
					// Update loss and evaulator
					EasyTrain.trainBatch(trainer, batch);
					// Update parameters
					trainer.step();
					batch.close();
				}
				// reset training and validation evaluators at end of epoch

				for (Batch batch : trainer.iterateDataset(testIter))
				{
					// Update loss and evaulator
					EasyTrain.validateBatch(trainer, batch);
					batch.close();
				}

				trainer.notifyListeners(listener -> listener.onEpoch(trainer));
				if (epoch % logInterval == 0)
				{
					epochCount[epoch / logInterval - 1] = epoch;
					trainLoss[epoch / logInterval - 1] = trainer.getTrainingResult().getEvaluations().get("train_loss");
					testLoss[epoch / logInterval - 1] = trainer.getTrainingResult().getEvaluations().get("validate_loss");
				}
			}
			System.out.println("Training complete...");
			model.close();

			String[] lossLabel = new String[trainLoss.length + testLoss.length];

			Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
			Arrays.fill(lossLabel, trainLoss.length, trainLoss.length + testLoss.length, "test loss");

			Table data = Table.create("Data").addColumns(DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)), DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, testLoss)), StringColumn.create("lossLabel", lossLabel));
			// Figure figure = LinePlot.create("Normal", data, "epochCount", "loss",
			// "lossLabel");
			Figure figure = LinePlot.create(fitting, data, "epochCount", "loss", "lossLabel");
			// set Y axis to log scale
			Axis yAxis = Axis.builder().type(Axis.Type.LOG).build();
			// Layout layout = Layout.builder("Normal").yAxis(yAxis).build();
			Layout layout = Layout.builder(fitting).yAxis(yAxis).build();
			figure.setLayout(layout);
			// render(figure, "text/html");
			Plot.show(figure);
		}
	}

	public static void train(float lambd) throws IOException, TranslateException
	{
		int nTrain = 20;
		// int nTest = 100;
		int numInputs = 200;
		int batchSize = 5;
		float trueB = 0.05f;

		try (NDManager manager = NDManager.newBaseManager())
		{
			InitParams initParams = new InitParams(manager, numInputs);

			NDArray trueW = manager.ones(new Shape(numInputs, 1));
			trueW = trueW.mul(0.01);

			DataPoints trainData = SyntheticData.create(manager, trueW, trueB, nTrain);
			ArrayDataset trainIter = MLPUtils.loadArray(trainData.getX(), trainData.getY(), batchSize, true);

			DataPoints testData = SyntheticData.create(manager, trueW, trueB, nTrain);
			// ArrayDataset testIter = PolynomialData.loadArray(testData.getX(),
			// testData.getY(), batchSize, false);

			NDList params = new NDList(initParams.getW(), initParams.getB());
			int numEpochs = Integer.getInteger("MAX_EPOCH", 100);
			float lr = 0.003f;

			double[] trainLoss = new double[(numEpochs / 5)];
			double[] testLoss = new double[(numEpochs / 5)];
			double[] epochCount = new double[(numEpochs / 5)];

			for (int epoch = 1; epoch <= numEpochs; epoch++)
			{
				for (Batch batch : trainIter.getData(manager))
				{
					NDArray X = batch.getData().head();
					NDArray y = batch.getLabels().head();

					NDArray w = params.get(0);
					NDArray b = params.get(1);

					try (GradientCollector gc = Engine.getInstance().newGradientCollector())
					{
						// The L2 norm penalty term has been added, and broadcasting
						// makes `l2Penalty(w)` a vector whose length is `batch_size`
						NDArray l = LNTraining.squaredLoss(LNTraining.linreg(X, w, b), y).add(l2Penalty(w).mul(lambd));
						gc.backward(l); // Compute gradient on l with respect to w and b
					}
					batch.close();
					LNTraining.sgd(params, lr, batchSize); // Update parameters using their gradient
				}

				if (epoch % 5 == 0)
				{
					NDArray testL = LNTraining.squaredLoss(LNTraining.linreg(testData.getX(), params.get(0), params.get(1)), testData.getY());
					NDArray trainL = LNTraining.squaredLoss(LNTraining.linreg(trainData.getX(), params.get(0), params.get(1)), trainData.getY());

					epochCount[epoch / 5 - 1] = epoch;
					trainLoss[epoch / 5 - 1] = trainL.mean().log10().getFloat();
					testLoss[epoch / 5 - 1] = testL.mean().log10().getFloat();
				}
			}
			System.out.println("l1 norm of w: " + params.get(0).abs().sum());

			String[] lossLabel = new String[trainLoss.length + testLoss.length];

			Arrays.fill(lossLabel, 0, testLoss.length, "test");
			Arrays.fill(lossLabel, testLoss.length, trainLoss.length + testLoss.length, "train");

			Table data = Table.create("Data").addColumns(DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)), DoubleColumn.create("loss", ArrayUtils.addAll(testLoss, trainLoss)), StringColumn.create("lossLabel", lossLabel));

			// render(LinePlot.create("", data, "epochCount", "loss",
			// "lossLabel"),"text/html");
			Plot.show(LinePlot.create("", data, "epochCount", "loss", "lossLabel"));
		}
	}

	public static void train_djl(float wd) throws IOException, TranslateException
	{
		int nTrain = 20;
		// int nTest = 100;
		int numInputs = 200;
		int batchSize = 5;
		float trueB = 0.05f;
		Loss l2Loss = Loss.l2Loss();

		try (NDManager manager = NDManager.newBaseManager())
		{
			InitParams initParams = new InitParams(manager, numInputs);

			NDArray trueW = manager.ones(new Shape(numInputs, 1));
			trueW = trueW.mul(0.01);

			DataPoints trainData = SyntheticData.create(manager, trueW, trueB, nTrain);
			ArrayDataset trainIter = MLPUtils.loadArray(trainData.getX(), trainData.getY(), batchSize, true);

			DataPoints testData = SyntheticData.create(manager, trueW, trueB, nTrain);
			// ArrayDataset testIter = PolynomialData.loadArray(testData.getX(),
			// testData.getY(), batchSize, false);

			NDList params = new NDList(initParams.getW(), initParams.getB());
			int numEpochs = Integer.getInteger("MAX_EPOCH", 100);
			float lr = 0.003f;

			double[] trainLoss = new double[(numEpochs / 5)];
			double[] testLoss = new double[(numEpochs / 5)];
			double[] epochCount = new double[(numEpochs / 5)];

			Tracker lrt = Tracker.fixed(lr);
			Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

			DefaultTrainingConfig config = new DefaultTrainingConfig(l2Loss).optOptimizer(sgd) // Optimizer (loss
																								// function)
					.optDevices(Engine.getInstance().getDevices(1)) // single CPU/GPU
					.addEvaluator(new Accuracy()) // Model Accuracy
					.addEvaluator(l2Loss).addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

			Model model = Model.newInstance("mlp");

			SequentialBlock net = new SequentialBlock();
			Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
			net.add(linearBlock);

			model.setBlock(net);
			Trainer trainer = model.newTrainer(config);

			trainer.initialize(new Shape(batchSize, 2));
			for (int epoch = 1; epoch <= numEpochs; epoch++)
			{
				for (Batch batch : trainer.iterateDataset(trainIter))
				{
					NDArray X = batch.getData().head();
					NDArray y = batch.getLabels().head();

					NDArray w = params.get(0);
					NDArray b = params.get(1);

					try (GradientCollector gc = Engine.getInstance().newGradientCollector())
					{
						// Minibatch loss in X and y
						NDArray l = LNTraining.squaredLoss(LNTraining.linreg(X, w, b), y).add(l2Penalty(w).mul(wd));
						gc.backward(l); // Compute gradient on l with respect to w and b
					}
					batch.close();
					LNTraining.sgd(params, lr, batchSize); // Update parameters using their gradient
				}

				if (epoch % 5 == 0)
				{
					NDArray testL = LNTraining.squaredLoss(LNTraining.linreg(testData.getX(), params.get(0), params.get(1)), testData.getY());
					NDArray trainL = LNTraining.squaredLoss(LNTraining.linreg(trainData.getX(), params.get(0), params.get(1)), trainData.getY());

					epochCount[epoch / 5 - 1] = epoch;
					trainLoss[epoch / 5 - 1] = trainL.mean().log10().getFloat();
					testLoss[epoch / 5 - 1] = testL.mean().log10().getFloat();
				}
			}
			System.out.println("l1 norm of w: " + params.get(0).abs().sum());

			String[] lossLabel = new String[trainLoss.length + testLoss.length];

			Arrays.fill(lossLabel, 0, testLoss.length, "test");
			Arrays.fill(lossLabel, testLoss.length, trainLoss.length + testLoss.length, "train");

			Table data = Table.create("Data").addColumns(DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, epochCount)), DoubleColumn.create("loss", ArrayUtils.addAll(testLoss, trainLoss)), StringColumn.create("lossLabel", lossLabel));

			// render(LinePlot.create("", data, "epochCount", "loss", "lossLabel"),
			// "text/html");
			Plot.show(LinePlot.create("", data, "epochCount", "loss", "lossLabel"));
		}
	}

	public static NDArray dropoutLayer(NDManager manager, NDArray X, float dropout)
	{
		// In this case, all elements are dropped out
		if (dropout == 1.0f)
		{
			return manager.zeros(X.getShape());
		}
		// In this case, all elements are kept
		if (dropout == 0f)
		{
			return X;
		}

		NDArray mask = manager.randomUniform(0f, 1.0f, X.getShape()).gt(dropout);
		return mask.toType(DataType.FLOAT32, false).mul(X).div(1.0f - dropout);
	}

	public static NDArray net(NDManager manager, NDArray X, NDArray W1, NDArray W2, NDArray W3, NDArray b1, NDArray b2, NDArray b3, boolean isTraining, float dropout1, float dropout2, int numInputs)
	{
		X = X.reshape(-1, numInputs);
		NDArray H1 = Activation.relu(X.dot(W1).add(b1));

		if (isTraining)
		{
			H1 = dropoutLayer(manager, H1, dropout1);
		}

		NDArray H2 = Activation.relu(H1.dot(W2).add(b2));
		if (isTraining)
		{
			H2 = dropoutLayer(manager, H2, dropout2);
		}

		return H2.dot(W3).add(b3);
	}

	public static void dropoutTrain() throws IOException, TranslateException
	{
		int numInputs = 784;
		int numOutputs = 10;
		int numHiddens1 = 256;
		int numHiddens2 = 256;

		int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
		float lr = 0.5f;
		int batchSize = 256;

		double[] trainLoss;
		double[] testAccuracy;
		double[] epochCount;
		double[] trainAccuracy;

		trainLoss = new double[numEpochs];
		trainAccuracy = new double[numEpochs];
		testAccuracy = new double[numEpochs];
		epochCount = new double[numEpochs];

		Loss loss = new SoftmaxCrossEntropyLoss();

		FashionMnist trainIter = FashionMnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		FashionMnist testIter = FashionMnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		trainIter.prepare();
		testIter.prepare();

		float epochLoss = 0f;
		float accuracyVal = 0f;

		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray W1 = manager.randomNormal(0, 0.01f, new Shape(numInputs, numHiddens1), DataType.FLOAT32);
			NDArray b1 = manager.zeros(new Shape(numHiddens1));
			NDArray W2 = manager.randomNormal(0, 0.01f, new Shape(numHiddens1, numHiddens2), DataType.FLOAT32);
			NDArray b2 = manager.zeros(new Shape(numHiddens2));
			NDArray W3 = manager.randomNormal(0, 0.01f, new Shape(numHiddens2, numOutputs), DataType.FLOAT32);
			NDArray b3 = manager.zeros(new Shape(numOutputs));

			NDList params = new NDList(W1, b1, W2, b2, W3, b3);

			for (NDArray param : params)
			{
				param.setRequiresGradient(true);
			}

			for (int epoch = 1; epoch <= numEpochs; epoch++)
			{
				// Iterate over dataset
				System.out.print("Running epoch " + epoch + "...... ");
				for (Batch batch : trainIter.getData(manager))
				{
					NDArray X = batch.getData().head();
					NDArray y = batch.getLabels().head();

					try (GradientCollector gc = Engine.getInstance().newGradientCollector())
					{
						NDArray yHat = MLPUtils.net(manager, X, W1, W2, W3, b1, b2, b3, true, 0.2f, 0.5f, numInputs); // net
																														// function
																														// call
						NDArray lossValue = loss.evaluate(new NDList(y), new NDList(yHat));
						NDArray l = lossValue.mul(batchSize);

						epochLoss += l.sum().getFloat();

						accuracyVal += LNTraining.accuracy(yHat, y);
						gc.backward(l); // gradient calculation
					}

					batch.close();
					LNTraining.sgd(params, lr, batchSize); // updater
				}

				trainLoss[epoch - 1] = epochLoss / trainIter.size();
				trainAccuracy[epoch - 1] = accuracyVal / trainIter.size();

				epochLoss = 0f;
				accuracyVal = 0f;

				for (Batch batch : testIter.getData(manager))
				{
					NDArray X = batch.getData().head();
					NDArray y = batch.getLabels().head();

					NDArray yHat = net(manager, X, W1, W2, W3, b1, b2, b3, false, 0.2f, 0.5f, numInputs); // net
																											// function
																											// call
					accuracyVal += LNTraining.accuracy(yHat, y);
				}
				testAccuracy[epoch - 1] = accuracyVal / testIter.size();
				epochCount[epoch - 1] = epoch;
				accuracyVal = 0f;
				System.out.println("Finished epoch " + epoch);
			}

			System.out.println("Finished training!");

			String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

			Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
			Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
			Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length, trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

			Table data = Table.create("Data").addColumns(DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))), DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))), StringColumn.create("lossLabel", lossLabel));

			// render(LinePlot.create("", data, "epochCount", "loss", "lossLabel"),
			// "text/html");
			Plot.show(LinePlot.create("", data, "epochCount", "loss", "lossLabel"));
		}
	}

	public static void conciseDropoutTrain(float dropout1, float dropout2) throws IOException, TranslateException
	{
		SequentialBlock net = new SequentialBlock();
		net.add(Blocks.batchFlattenBlock(784));
		net.add(Linear.builder().setUnits(256).build());
		net.add(Activation::relu);
		net.add(Dropout.builder().optRate(dropout1).build());
		net.add(Linear.builder().setUnits(256).build());
		net.add(Activation::relu);
		net.add(Dropout.builder().optRate(dropout2).build());
		net.add(Linear.builder().setUnits(10).build());
		net.setInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT);

		Map<String, double[]> evaluatorMetrics = new HashMap<>();

		Tracker lrt = Tracker.fixed(0.5f);
		Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

		Loss loss = Loss.softmaxCrossEntropyLoss();

		DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
				.optDevices(Engine.getInstance().getDevices(1)) // single GPU
				.addEvaluator(new Accuracy()) // Model Accuracy
				.addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

		int batchSize = 256;
		int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

		FashionMnist trainIter = FashionMnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		FashionMnist testIter = FashionMnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batchSize, true).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		trainIter.prepare();
		testIter.prepare();

		double[] epochCount = new double[numEpochs];
		int index = 0;
		try (Model model = Model.newInstance("mlp"))
		{
			model.setBlock(net);
			try (Trainer trainer = model.newTrainer(config))
			{
				trainer.initialize(new Shape(1, 784));
				trainer.setMetrics(new Metrics());

				EasyTrain.fit(trainer, numEpochs, trainIter, testIter);

				Metrics metrics = trainer.getMetrics();

				trainer.getEvaluators().stream().forEach(evaluator ->
				{
					evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
					evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream().mapToDouble(x -> x.getValue().doubleValue()).toArray());
				});
			}
			
			// The below two lines are added by me. 08/18/2024, Bing Li
			epochCount[index] = index + 1;
			index++;
		}
		
		double[] trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
		double[] trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
		double[] testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

		String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

		Arrays.fill(lossLabel, 0, trainLoss.length, "test acc");
		Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
		Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
		                trainLoss.length + testAccuracy.length + trainAccuracy.length, "train loss");

		Table data = Table.create("Data").addColumns(
		    DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
		    DoubleColumn.create("loss", ArrayUtils.addAll(testAccuracy , ArrayUtils.addAll(trainAccuracy, trainLoss))),
		    StringColumn.create("lossLabel", lossLabel)
		);

//		render(LinePlot.create("", data, "epochCount", "loss", "lossLabel"),"text/html");
		Plot.show(LinePlot.create("", data, "epochCount", "loss", "lossLabel"));
	}
}
