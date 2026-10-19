package ai.greatfree.rnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import ai.greatfree.utils.Functions.TriFunction;
import ai.greatfree.utils.RNNModelScratch;
import ai.greatfree.utils.Vocab;

/**
 * Created: 08/28/2024, Bing Li
 */
final class RNN
{
	public static void main(String[] args) throws IOException, Exception
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			int batchSize = 32;
			int numSteps = 35;
			Pair<ArrayList<NDList>, Vocab> timeMachine = RNNUtils.loadDataTimeMachine(batchSize, numSteps, false, 10000, manager);
//			Pair<ArrayList<NDList>, Vocab> timeMachine = RNNUtils.loadDataTimeMachine(batchSize, numSteps, false, 20000, manager);
//			Pair<ArrayList<NDList>, Vocab> timeMachine = RNNUtils.loadDataTimeMachine(batchSize, numSteps, false, 200, manager);
			List<NDList> trainIter = timeMachine.getKey();
			Vocab vocab = timeMachine.getValue();

			/*
			for (int i = 0; i < vocab.length(); i++)
			{
				System.out.println(i + ") vocab = " + vocab.toToken(i));
			}
			
			System.out.println("vocab size = " + vocab.length());
			
			int index = 0;
			for (NDList entry : trainIter)
			{
				for (NDArray arr : entry.getResourceNDArrays())
				{
					System.out.println(index++ + ") NDArray size = " + arr.encode().length);
				}
			}

			System.out.println("Press Enter to continue ...");
			Tools.INPUT.nextLine();
			*/

			NDArray hots = manager.create(new int[] { 0, 2 }).oneHot(vocab.length());
			System.out.println(hots);

			NDArray X = manager.arange(10).reshape(new Shape(2, 5));
			Shape s = X.transpose().oneHot(28).getShape();
			System.out.println(s);

			int numHiddens = 512;
			TriFunction<Integer, Integer, Device, NDList> getParamsFn = (a, b, c) -> RNNUtils.getParams(a, b, c, manager);
			TriFunction<Integer, Integer, Device, NDList> initRNNStateFn = (a, b, c) -> RNNUtils.initRNNState(a, b, c, manager);
			TriFunction<NDArray, NDList, NDList, Pair<NDArray, NDList>> rnnFn = (a, b, c) -> RNNUtils.rnn(a, b, c);

			Device device = manager.getDevice();

			RNNModelScratch net = new RNNModelScratch(vocab.length(), numHiddens, device, getParamsFn, initRNNStateFn, rnnFn);
			NDList state = net.beginState((int) X.getShape().getShape()[0], device);
			Pair<NDArray, NDList> pairResult = net.forward(X.toDevice(device, false), state);
			NDArray Y = pairResult.getKey();
			NDList newState = pairResult.getValue();
			System.out.println(Y.getShape());
			System.out.println(newState.get(0).getShape());

			String prediction = RNNUtils.predict("time traveller ", 10, net, vocab, manager.getDevice(), manager);
			System.out.println(prediction);

			int numEpochs = Integer.getInteger("MAX_EPOCH", 500);
//			int numEpochs = Integer.getInteger("MAX_EPOCH", 100);

			int lr = 1;
			/*
			for (NDList entry : trainIter)
			{
//				System.out.println(entry);
				for (NDArray dn : entry.getResourceNDArrays())
				{
					System.out.println("dn length = " + dn.encode().length);
					System.out.println("dn = " + dn);
				}
			}
			System.out.println("Press Enter to continue ...");
			Tools.INPUT.nextLine();
			*/
			RNNUtils.train(net, trainIter, vocab, lr, numEpochs, manager.getDevice(), false, manager);
		}
	}
}

