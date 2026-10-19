package ai.greatfree.linearregression;

import java.io.IOException;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.greatfree.utils.StopWatch;

/**
 * Created: 08/14/2024, Bing Li
 */
final class GetDataSet
{
	public static void main(String[] args) throws IOException, TranslateException
	{
		int batchSize = 256;
		boolean randomShuffle = true;

		FashionMnist mnistTrain = FashionMnist.builder().optUsage(Dataset.Usage.TRAIN).setSampling(batchSize, randomShuffle).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		FashionMnist mnistTest = FashionMnist.builder().optUsage(Dataset.Usage.TEST).setSampling(batchSize, randomShuffle).optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE)).build();

		mnistTrain.prepare();
		mnistTest.prepare();

		try (NDManager manager = NDManager.newBaseManager())
		{
			System.out.println(mnistTrain.size());
			System.out.println(mnistTest.size());

			final int SCALE = 4;
			final int WIDTH = 28;
			final int HEIGHT = 28;
			/*
			 * The images are not shown successfully! 08/14/2024, Bing Li
			 */
			MNISTData.showImages(mnistTrain, 6, WIDTH, HEIGHT, SCALE, manager);

			StopWatch stopWatch = new StopWatch();
			stopWatch.start();
			for (Batch batch : mnistTrain.getData(manager))
			{
				NDArray x = batch.getData().head();
				System.out.println(x);

				System.out.println("--------------------");
				NDArray y = batch.getLabels().head();
				System.out.println(y);
			}
			System.out.println(String.format("%.2f sec", stopWatch.stop()));
		}
	}
}
