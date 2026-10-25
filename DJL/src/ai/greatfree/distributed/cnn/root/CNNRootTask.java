package ai.greatfree.distributed.cnn.root;

import java.io.IOException;
import java.util.Calendar;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;

import ai.greatfree.distributed.cnn.message.ChildPathsRequest;
import ai.greatfree.distributed.cnn.message.ChildPathsResponse;
import ai.greatfree.distributed.cnn.message.CNNAppID;
import edu.greatfree.cluster.message.ChildRequest;
import edu.greatfree.cluster.message.ClusterNotification;
import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.ClusterResponse;
import edu.greatfree.cluster.message.RootResponse;
import edu.greatfree.cluster.root.RootTask;
import edu.greatfree.cluster.root.UnaryRoot;
import edu.greatfree.framework.cluster.multicast.message.ClusterAppID;

/**
 * Created: 08/29/2024, Bing Li
 */
final class CNNRootTask extends RootTask
{
	private final static Logger log = Logger.getLogger("ai.greatfree.distributed.rnn.root");

	@Override
	public RootResponse processChildRequest(ChildRequest request)
	{
		switch (request.getAppID())
		{
			case CNNAppID.CHILD_PATHS_REQUEST:
				log.info("CHILD_PATHS_REQUEST received @" + Calendar.getInstance().getTime());
				ChildPathsRequest cpr = (ChildPathsRequest)request;
				return new ChildPathsResponse(Paths.ROOT().getPath(cpr.getTrainingDataPath(), cpr.getTrainedDataPath()));
		}
		return null;
	}

	@Override
	public void processNotification(ClusterNotification notification)
	{
		switch (notification.getAppID())
		{
			case ClusterAppID.SHUTDOWN_ROOT_NOTIFICATION:
				log.info("SHUTDOWN_ROOT_NOTIFICATION received @" + Calendar.getInstance().getTime());
				try
				{
					UnaryRoot.CLUSTER().stop();
				}
				catch (ClassNotFoundException | IOException | InterruptedException | RemoteReadException e)
				{
					e.printStackTrace();
				}
				break;
		}
		
	}

	@Override
	public ClusterResponse processRequest(ClusterRequest request)
	{
		// TODO Auto-generated method stub
		return null;
	}
}
