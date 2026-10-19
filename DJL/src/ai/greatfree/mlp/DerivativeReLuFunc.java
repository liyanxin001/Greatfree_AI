package ai.greatfree.mlp;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import ai.djl.training.GradientCollector;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Created: 08/15/2024, Bing Li
 */
final class DerivativeReLuFunc
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(-8.0f, 8.0f, 0.1f);
			System.out.println(x);
			x.setRequiresGradient(true);

			NDArray y;
			try (GradientCollector collector = Engine.getInstance().newGradientCollector())
			{
				y = Activation.relu(x);
				System.out.println("--------------------");
				System.out.println(y);
				collector.backward(y);
			}

			NDArray res = x.getGradient();
			System.out.println("--------------------");
			System.out.println(res);
			float[] X = x.toFloatArray();
			float[] Y = res.toFloatArray();

			Table data = Table.create("Data").addColumns(FloatColumn.create("X", X), FloatColumn.create("grad of relu", Y));
//			Plot.show(LinePlot.create("", data, "x", "grad of relu"), "text/html");
			Plot.show(LinePlot.create("", data, "x", "grad of relu"));
		}
	}
}
