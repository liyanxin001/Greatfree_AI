package ai.greatfree.distributed.cnn.message;

import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.RequestType;

/**
 * Created: 08/29/2024, Bing Li
 */
public class InitRequest extends ClusterRequest
{
	private static final long serialVersionUID = -3435973025978637522L;

	public InitRequest()
	{
		super(RequestType.BROADCAST_REQUEST, CNNAppID.INIT_REQUEST);
	}
}

