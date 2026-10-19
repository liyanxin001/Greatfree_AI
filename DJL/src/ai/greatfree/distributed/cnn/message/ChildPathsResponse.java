package ai.greatfree.distributed.cnn.message;

import ai.greatfree.distributed.rnn.message.ChildPath;
import edu.greatfree.cluster.message.RootResponse;

/**
 * Created: 08/29/2024, Bing Li
 */
public class ChildPathsResponse extends RootResponse
{
	private static final long serialVersionUID = 4816435746865174335L;

	private ChildPath path;

	public ChildPathsResponse(ChildPath path)
	{
		super(CNNAppID.CHILD_PATHS_RESPONSE);
		this.path = path;
	}
	
	public ChildPath getPath()
	{
		return this.path;
	}
}
