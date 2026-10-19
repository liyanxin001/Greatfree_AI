package ai.greatfree.computation;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

/**
 * Created: 08/19/2024, Bing Li
 */
final class CustomParamsLayer
{
	public static void main(String[] args) throws TranslateException
	{
		// 5 units in -> 3 units out
		MyLinear linear = new MyLinear(3, 5);
		var params = linear.getParameters();
		for (Pair<String, Parameter> param : params)
		{
			System.out.println(param.getKey());
		}

		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray input = manager.randomUniform(0, 1, new Shape(2, 5));
			linear.initialize(manager, DataType.FLOAT32, input.getShape());
			Model model = Model.newInstance("my-linear");
			model.setBlock(linear);
			Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator());
			System.out.println(predictor.predict(new NDList(input)).singletonOrThrow());

			System.out.println("------------------------");
			input = manager.randomUniform(0, 1, new Shape(2, 64));

			SequentialBlock net = new SequentialBlock();
			net.add(new MyLinear(8, 64)); // 64 units in -> 8 units out
			net.add(new MyLinear(1, 8)); // 8 units in -> 1 unit out
			net.initialize(manager, DataType.FLOAT32, input.getShape());

			model = Model.newInstance("lin-reg-custom");
			model.setBlock(net);

			predictor = model.newPredictor(new NoopTranslator());
			System.out.println(predictor.predict(new NDList(input)).singletonOrThrow());
		}
	}
}

