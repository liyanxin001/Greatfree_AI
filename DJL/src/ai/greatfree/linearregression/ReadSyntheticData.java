package ai.greatfree.linearregression;

import java.io.IOException;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.TranslateException;

/**
 * Created: 08/13/2024, Bing Li
 */
final class ReadSyntheticData
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray trueW = manager.create(new float[] { 2, -3.4f });
			float trueB = 4.2f;

			DataPoints dp = SyntheticData.create(manager, trueW, trueB, 1000);
			NDArray features = dp.getX();
			NDArray labels = dp.getY();

			System.out.printf("features: [%f, %f]\n", features.get(0).getFloat(0), features.get(0).getFloat(1));
			System.out.println("label: " + labels.getFloat(0));

			int batchSize = 10;

			ArrayDataset dataset = new ArrayDataset.Builder().setData(features) // Set the Features
					.optLabels(labels) // Set the Labels
					.setSampling(batchSize, false) // set the batch size and random sampling to false
					.build();

			Batch batch = dataset.getData(manager).iterator().next();
			// Call head() to get the first NDArray
			NDArray x = batch.getData().head();
			NDArray y = batch.getLabels().head();
			System.out.println(x);
			System.out.println(y);
			// Don't forget to close the batch!
			batch.close();
		}
	}
}

