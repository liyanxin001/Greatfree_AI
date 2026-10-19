package ai.greatfree.linearregression;

import java.io.IOException;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

/**
 * The code can be re-implemented in a distributed environment with GreatFree. 08/13/2024, Bing Li
 * 
 * Created: 08/13/2024, Bing Li
 */
final class InitializeModelParamsAndTraining
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray trueW = manager.create(new float[] { 2, -3.4f });
			System.out.println("trueW = " + trueW);

			float trueB = 4.2f;
			System.out.println("------------------------");
			DataPoints dp = SyntheticData.create(manager, trueW, trueB, 1000);
			NDArray features = dp.getX();
			System.out.println("features = " + features);
			
			System.out.println("------------------------");
			NDArray labels = dp.getY();
			System.out.println("labels = " + labels);

			int batchSize = 10;
			ArrayDataset dataset = new ArrayDataset.Builder()
					.setData(features) // Set the Features
					.optLabels(labels) // Set the Labels
					.setSampling(batchSize, false) // set the batch size and random sampling to false
					.build();

			Batch batch = dataset.getData(manager).iterator().next();

			// Call head() to get the first NDArray
			System.out.println("------------------------");
			NDArray x = batch.getData().head();
			System.out.println("x = " + x);

			System.out.println("------------------------");
			NDArray y = batch.getLabels().head();
			System.out.println("y = " + y);

			// Don't forget to close the batch!
			batch.close();

			System.out.println("------------------------");
			NDArray w = manager.randomNormal(0, 0.01f, new Shape(2, 1), DataType.FLOAT32);
			System.out.println("w = " + w);

			System.out.println("------------------------");
			NDArray b = manager.zeros(new Shape(1));
			System.out.println("b = " + b);

			System.out.println("------------------------");
			NDList params = new NDList(w, b);
			System.out.println(params);

			/*
			 * Training is performed as follows. 08/13/2024, Bing Li
			 */
			float lr = 0.03f; // Learning Rate
			int numEpochs = 3; // Number of Iterations

			// Attach Gradients
			for (NDArray param : params)
			{
				param.setRequiresGradient(true);
			}

			for (int epoch = 0; epoch < numEpochs; epoch++)
			{
				// Assuming the number of examples can be divided by the batch size, all
				// the examples in the training dataset are used once in one epoch
				// iteration. The features and tags of minibatch examples are given by X
				// and y respectively.
				for (Batch entry : dataset.getData(manager))
				{
					NDArray s = entry.getData().head();
					NDArray t = entry.getLabels().head();

					try (GradientCollector gc = Engine.getInstance().newGradientCollector())
					{
						// Minibatch loss in X and y
						NDArray l = LNTraining.squaredLoss(LNTraining.linreg(s, params.get(0), params.get(1)), t);
						gc.backward(l); // Compute gradient on l with respect to w and b
					}
					LNTraining.sgd(params, lr, batchSize); // Update parameters using their gradient
					entry.close();
				}
				NDArray trainL = LNTraining.squaredLoss(LNTraining.linreg(features, params.get(0), params.get(1)), labels);
				System.out.printf("epoch %d, loss %f\n", epoch + 1, trainL.mean().getFloat());
			}

			float[] v = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
			System.out.println(String.format("Error in estimating w: [%f, %f]", v[0], v[1]));
			System.out.println(String.format("Error in estimating b: %f", trueB - params.get(1).getFloat()));
		}
	}
}
