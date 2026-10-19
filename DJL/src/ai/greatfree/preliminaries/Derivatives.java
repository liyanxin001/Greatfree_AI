package ai.greatfree.preliminaries;

import java.util.function.Function;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;

/**
 * Created: 08/11/2024, Bing Li
 */
final class Derivatives
{
	public static void main(String[] args)
	{
		Function<Double, Double> f = x -> 2 * x - 3;
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.arange(0f, 3f, 0.1f, DataType.FLOAT64);
			double[] x = X.toDoubleArray();

			double[] fx = new double[x.length];
			for (int i = 0; i < x.length; i++)
			{
				fx[i] = f.apply(x[i]);
			}

			double[] fg = new double[x.length];
			for (int i = 0; i < x.length; i++)
			{
				fg[i] = 2 * x[i] - 3;
			}

			Figure figure = Plotly.plotLineAndSegment(x, fx, fg, "f(x)", "Tangent line(x=1)", "x", "f(x)", 700, 500);
			Plot.show(figure);
		}
	}
}
