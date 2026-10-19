package ai.greatfree.distributed.cnn.child;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.util.Calendar;
import java.util.List;
import java.util.logging.Logger;

import javax.imageio.ImageIO;

import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.FileManager;

import ai.djl.util.Pair;
import ai.greatfree.distributed.cnn.child.app.ChildProfile;
import ai.greatfree.distributed.cnn.child.app.CNNTrainer;
import ai.greatfree.distributed.cnn.message.CNNAppID;
import ai.greatfree.distributed.cnn.message.CollectGradientsRequest;
import ai.greatfree.distributed.cnn.message.CollectGradientsResponse;
import ai.greatfree.distributed.cnn.message.DistributeTrainingDataRequest;
import ai.greatfree.distributed.cnn.message.DistributeTrainingDataResponse;
import ai.greatfree.distributed.cnn.message.InitRequest;
import ai.greatfree.distributed.cnn.message.InitResponse;
import ai.greatfree.distributed.cnn.message.PredictRequest;
import ai.greatfree.distributed.cnn.message.PredictResponse;
import ai.greatfree.distributed.cnn.utils.CNNTools;
import edu.greatfree.cluster.child.ChildTask;
import edu.greatfree.cluster.message.ClusterNotification;
import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.ClusterResponse;
import edu.greatfree.cluster.message.InterChildrenNotification;
import edu.greatfree.cluster.message.InterChildrenRequest;
import edu.greatfree.cluster.message.InterStaticNotification;
import edu.greatfree.cluster.message.InterStaticRequest;
import edu.greatfree.cluster.message.IntercastNotification;
import edu.greatfree.cluster.message.IntercastRequest;
import edu.greatfree.cluster.message.IsReplicaAvailableRequest;
import edu.greatfree.cluster.message.IsReplicaAvailableResponse;
import edu.greatfree.cluster.message.Replica;
import edu.greatfree.cluster.message.ReplicationRequest;
import edu.greatfree.cluster.message.ReplicationResponse;
import edu.greatfree.framework.cluster.multicast.message.ClusterAppID;
import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 10/16/2026
 * CNN counterpart to RNNChildTask.
 */
final class CNNChildTask extends ChildTask
{
    private final static Logger log = Logger.getLogger("ai.greatfree.distributed.cnn.child");

    @Override
    public InterStaticNotification assignDestinations(InterStaticNotification notification)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public InterStaticRequest assignDestinations(InterStaticRequest request)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IntercastNotification assignDestinations(IntercastNotification notification)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IntercastRequest assignDestinations(IntercastRequest request)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public IsReplicaAvailableResponse isReplicaAvailable(IsReplicaAvailableRequest request)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public InterChildrenNotification prepareNotification(IntercastNotification notification)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Replica prepareReplication(int arg0)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public InterChildrenRequest prepareRequest(IntercastRequest request)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void processFailedReplication(ReplicationResponse response)
    {
        // TODO Auto-generated method stub
    }

    // ============================================================
    //  NOTIFICATIONS
    // ============================================================

    @Override
    public void processNotification(ClusterNotification notification)
    {
        switch (notification.getAppID())
        {
            case CNNAppID.START_TRAINING_NOTIFICATION:
                log.info("START_TRAINING_NOTIFICATION received @" + Calendar.getInstance().getTime());
                try
                {
                    CNNTrainer.CHILD().train();
                }
                catch (ClassNotFoundException | RemoteReadException | IOException | InterruptedException e)
                {
                    e.printStackTrace();
                }
                break;

            case ClusterAppID.SHUTDOWN_CHILDREN_NOTIFICATION:
                log.info("SHUTDOWN_CHILDREN_NOTIFICATION received @" + Calendar.getInstance().getTime());
                try
                {
                    CNNChild.CLUSTER().stop();
                }
                catch (ClassNotFoundException | IOException | InterruptedException | RemoteReadException e)
                {
                    e.printStackTrace();
                }
                break;
        }
    }

    @Override
    public void processNotification(InterChildrenNotification notification)
    {
        // TODO Auto-generated method stub
    }

    @Override
    public void processNotification(InterChildrenNotification notification, List<String> distinationKeys)
    {
        // TODO Auto-generated method stub
    }

    // ============================================================
    //  REQUESTS  — four cases differ from the RNN version
    // ============================================================

    @Override
    public MulticastResponse processRequest(ClusterRequest request)
    {
        switch (request.getAppID())
        {
            // ---------------------------------------------------------
            //  CASE 1: Distribute training data (reshaped)
            //  RNN: write lines of text to a file
            //  CNN: decode image bytes and write <class>/img_N.png
            // ---------------------------------------------------------
            case CNNAppID.DISTRIBUTE_TRAINING_DATA_REQUEST:
            {
                log.info("DISTRIBUTE_TRAINING_DATA_REQUEST received @" + Calendar.getInstance().getTime());
                DistributeTrainingDataRequest dtdr = (DistributeTrainingDataRequest) request;
                try
                {
                    String rootDir = ChildProfile.CHILD().getTrainingDataPath();
                    String baseDir = rootDir + dtdr.getFileName() + "/";

                    if (!FileManager.isDirExisted(baseDir))
                    {
                        FileManager.makeDir(baseDir);
                    }

                    List<byte[]> images     = dtdr.getImages();
                    List<Integer> labels    = dtdr.getLabels();
                    List<String> classNames = dtdr.getClassNames();

                    for (int i = 0; i < images.size(); i++)
                    {
                        String className = classNames.get(labels.get(i));
                        String classDir  = baseDir + className + "/";
                        if (!FileManager.isDirExisted(classDir))
                        {
                            FileManager.makeDir(classDir);
                        }

                        File out = new File(classDir, "img_" + i + ".png");
                        java.nio.file.Files.write(out.toPath(), images.get(i));
                    }

                    ChildProfile.CHILD().setTrainingDataPath(baseDir);
                    log.info("Wrote " + images.size() + " images across "
                             + classNames + " to " + baseDir);
                }
                catch (IOException e)
                {
                    e.printStackTrace();
                    return new DistributeTrainingDataResponse(false, dtdr.getChildKey(), dtdr.getCollaboratorKey());
                }
                return new DistributeTrainingDataResponse(true, dtdr.getChildKey(), dtdr.getCollaboratorKey());
            }

            // ---------------------------------------------------------
            //  CASE 2: Init (identical to RNN, just CNNTrainer)
            // ---------------------------------------------------------
            case CNNAppID.INIT_REQUEST:
            {
                log.info("INIT_REQUEST received @" + Calendar.getInstance().getTime());
                InitRequest ir = (InitRequest) request;
                try
                {
                    CNNTrainer.CHILD().init();
                }
                catch (Exception e)
                {
                    e.printStackTrace();
                    return new InitResponse(CNNChild.CLUSTER().getChildKey(), false, ir.getCollaboratorKey());
                }
                return new InitResponse(CNNChild.CLUSTER().getChildKey(), true, ir.getCollaboratorKey());
            }

            // ---------------------------------------------------------
            //  CASE 3: Collect gradients (type name only)
            //  RNN: RNNTools.getGradients(RNNTrainer.CHILD().getRNN())
            //  CNN: CNNTools.getGradients(CNNTrainer.CHILD().getCNN())
            // ---------------------------------------------------------
            case CNNAppID.COLLECT_GRADIENTS_REQUEST:
            {
                log.info("COLLECT_GRADIENTS_REQUEST received @" + Calendar.getInstance().getTime());
                CollectGradientsRequest cgr = (CollectGradientsRequest) request;
                return new CollectGradientsResponse(
                        CNNTools.getGradients(CNNTrainer.CHILD().getCNN()),
                        cgr.getCollaboratorKey());
            }

            // ---------------------------------------------------------
            //  CASE 4: Predict (reshaped)
            //  RNN: pass a prefix string, return generated text
            //  CNN: decode image bytes, return (label, confidence)
            // ---------------------------------------------------------
            case CNNAppID.PREDICT_REQUEST:
            {
                log.info("PREDICT_REQUEST received @" + Calendar.getInstance().getTime());
                PredictRequest pr = (PredictRequest) request;
                try
                {
                    BufferedImage img = ImageIO.read(new ByteArrayInputStream(pr.getImageBytes()));
                    if (img == null)
                    {
                        return new PredictResponse("ERROR: undecodable image",
                                                   0.0f, pr.getCollaboratorKey());
                    }

                    Pair<String, Float> result = CNNTrainer.CHILD().predict(img);
                    return new PredictResponse(result.getKey(),
                                               result.getValue(),
                                               pr.getCollaboratorKey());
                }
                catch (IOException e)
                {
                    e.printStackTrace();
                    return new PredictResponse("ERROR", 0.0f, pr.getCollaboratorKey());
                }
            }
        }
        return null;
    }

    @Override
    public List<MulticastResponse> processRequest(InterChildrenRequest request)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public List<MulticastResponse> processRequest(InterChildrenRequest request, List<String> distinationKeys)
    {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void processResponse(ClusterResponse response)
    {
        // TODO Auto-generated method stub
    }

    @Override
    public ReplicationResponse replicate(ReplicationRequest request)
    {
        // TODO Auto-generated method stub
        return null;
    }
}
