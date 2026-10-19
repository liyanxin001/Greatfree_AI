package ai.greatfree.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Created: 08/16/2024, Bing Li
 */
final class TanhFunc
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(-8.0f, 8.0f, 0.1f);
			System.out.println(x);
			x.setRequiresGradient(true);
			
			System.out.println("--------------------");
			NDArray y = Activation.tanh(x);
			System.out.println(y);

			float[] X = x.toFloatArray();
			float[] Y = y.toFloatArray();

			Table data = Table.create("Data").addColumns(FloatColumn.create("X", X), FloatColumn.create("tanh(x)", Y));
			Plot.show(LinePlot.create("", data, "x", "tanh(X)"));
		}
	}
}

