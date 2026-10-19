package ai.greatfree.linearregression;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.greatfree.utils.Accumulator;
import ai.greatfree.utils.Animator;
import ai.greatfree.utils.FashionMnistUtils;

/**
 * Created: 08/13/2024, Bing Li
 */
public final class LNTraining
{
	// Define the model. 08/13/2024, Bing Li
	public static NDArray linreg(NDArray x, NDArray w, NDArray b)
	{
		return x.dot(w).add(b);
	}

	// Define the loss function. 08/13/2024, Bing Li
	public static NDArray squaredLoss(NDArray yHat, NDArray y)
	{
		return (yHat.sub(y.reshape(yHat.getShape()))).mul((yHat.sub(y.reshape(yHat.getShape())))).div(2);
	}

	// Define the optimization algorithm. 08/13/2024, Bing Li
	public static void sgd(NDList params, float lr, int batchSize)
	{
		for (int i = 0; i < params.size(); i++)
		{
			NDArray param = params.get(i);
			// Update param
			// param = param - param.gradient * lr / batchSize
			param.subi(param.getGradient().mul(lr).div(batchSize));
		}
	}

	public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle)
	{
		return new ArrayDataset.Builder().setData(features) // set the features
				.optLabels(labels) // set the labels
				.setSampling(batchSize, shuffle) // set the batch size and random sampling
				.build();
	}

	public static NDArray softmax(NDArray X)
	{
		NDArray Xexp = X.exp();
		NDArray partition = Xexp.sum(new int[] { 1 }, true);
		return Xexp.div(partition); // The broadcast mechanism is applied here
	}

	public static float accuracy(NDArray yHat, NDArray y)
	{
		// Check size of 1st dimension greater than 1
		// to see if we have multiple samples
		if (yHat.getShape().size(1) > 1)
		{
			// Argmax gets index of maximum args for given axis 1
			// Convert yHat to same dataType as y (int32)
			// Sum up number of true entries
			return yHat.argMax(1).toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false)).sum().toType(DataType.FLOAT32, false).getFloat();
		}
		return yHat.toType(DataType.INT32, false).eq(y.toType(DataType.INT32, false)).sum().toType(DataType.FLOAT32, false).getFloat();
	}

	public static float evaluateAccuracy(UnaryOperator<NDArray> net, Iterable<Batch> dataIterator)
	{
		Accumulator metric = new Accumulator(2); // numCorrectedExamples, numExamples
		Batch batch = dataIterator.iterator().next();
		NDArray X = batch.getData().head();
		NDArray y = batch.getLabels().head();
		metric.add(new float[] { accuracy(net.apply(X), y), (float) y.size() });
		batch.close();
		return metric.get(0) / metric.get(1);
	}

	@FunctionalInterface
	public static interface ParamConsumer
	{
		void accept(NDList params, float lr, int batchSize);
	}

	// public static float[] trainEpochCh3(UnaryOperator<NDArray> net,
	// Iterable<Batch> trainIter, BinaryOperator<NDArray> loss, ParamConsumer
	// updater, NDList params, int numInputs, float lr)
	public static float[] trainEpochCh3(UnaryOperator<NDArray> net, Iterable<Batch> trainIter, BinaryOperator<NDArray> loss, ParamConsumer updater)
	{
		Accumulator metric = new Accumulator(3); // trainLossSum, trainAccSum, numExamples

		// Attach Gradients
		for (NDArray param : SoftMaxProfile.TRAIN().getParams())
		{
			param.setRequiresGradient(true);
		}

		for (Batch batch : trainIter)
		{
			NDArray X = batch.getData().head();
			NDArray y = batch.getLabels().head();
			X = X.reshape(new Shape(-1, SoftMaxProfile.TRAIN().getNumInputs()));

			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				// Minibatch loss in X and y
				NDArray yHat = net.apply(X);
				NDArray l = loss.apply(yHat, y);
				gc.backward(l); // Compute gradient on l with respect to w and b
				metric.add(new float[] { l.sum().toType(DataType.FLOAT32, false).getFloat(), LNTraining.accuracy(yHat, y), (float) y.size() });
				gc.close();
			}
			updater.accept(SoftMaxProfile.TRAIN().getParams(), SoftMaxProfile.TRAIN().getLR(), batch.getSize()); // Update
																													// parameters
																													// using
																													// their
																													// gradient
			batch.close();
		}
		// Return trainLoss, trainAccuracy
		return new float[] { metric.get(0) / metric.get(2), metric.get(1) / metric.get(2) };
	}

	// public static void trainCh3(NDManager manager, UnaryOperator<NDArray> net,
	// Dataset trainDataset, Dataset testDataset, BinaryOperator<NDArray> loss, int
	// numEpochs, ParamConsumer updater, NDList params, int numInputs, float lr)
	// throws IOException, TranslateException
	public static void trainCh3(UnaryOperator<NDArray> net, Dataset trainDataset, Dataset testDataset, BinaryOperator<NDArray> loss, int numEpochs, ParamConsumer updater) throws IOException, TranslateException
	{
		Animator animator = new Animator();
		for (int i = 1; i <= numEpochs; i++)
		{
			// float[] trainMetrics = trainEpochCh3(net,
			// trainDataset.getData(SoftMaxProfile.TRAIN().getManager()), loss, updater,
			// SoftMaxProfile.TRAIN().getParams(), SoftMaxProfile.TRAIN().getNumInputs(),
			// SoftMaxProfile.TRAIN().getLR());
			float[] trainMetrics = trainEpochCh3(net, trainDataset.getData(SoftMaxProfile.TRAIN().getManager()), loss, updater);
			float accuracy = LNTraining.evaluateAccuracy(net, testDataset.getData(SoftMaxProfile.TRAIN().getManager()));
			float trainAccuracy = trainMetrics[1];
			float trainLoss = trainMetrics[0];

			animator.add(i, accuracy, trainAccuracy, trainLoss);
			System.out.printf("Epoch %d: Test Accuracy: %f\n", i, accuracy);
			System.out.printf("Train Accuracy: %f\n", trainAccuracy);
			System.out.printf("Train Loss: %f\n", trainLoss);
		}
	}

	public static BufferedImage predictCh3(UnaryOperator<NDArray> net, ArrayDataset dataset, int number, NDManager manager) throws IOException, TranslateException
	{
		int[] predLabels = new int[number];

		Batch batch = dataset.getData(manager).iterator().next();
		NDArray X = batch.getData().head();
		int[] yHat = net.apply(X).argMax(1).toType(DataType.INT32, false).toIntArray();
		for (int i = 0; i < number; i++)
		{
			predLabels[i] = yHat[i];
			System.out.println("label = " + predLabels[i]);
		}
		return FashionMnistUtils.showImages(dataset, predLabels, 28, 28, 4, manager);
	}
}
