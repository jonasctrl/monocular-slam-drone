#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <vector>

class Navigator
{
public:
    Navigator() : nh_("~"), octree_(nullptr), resolution_(2), // Increase the step size here
                  current_position_(0.0, 0.0, 0.0), current_orientation_(0.0, 0.0, 0.0, 1.0),
                  octomap_received_(false), pose_received_(false)
    {
        octomap_sub_ = nh_.subscribe("/octomap_binary", 1, &Navigator::octomapCallback, this);
        pose_sub_ = nh_.subscribe("/orb_slam3/camera_pose", 1, &Navigator::poseCallback, this);
        goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10);

        nh_.param<double>("resolution", resolution_, 2); // Adjusted step size
        nh_.param<double>("goal_x", goal_x_, 1.0);
        nh_.param<double>("goal_y", goal_y_, 2.0);
        nh_.param<double>("goal_z", goal_z_, 0.5);

        ROS_INFO("Navigator initialized with goal (%.2f, %.2f, %.2f) and step size %.2f",
                 goal_x_, goal_y_, goal_z_, resolution_);
    }

    ~Navigator()
    {
        if (octree_)
        {
            delete octree_;
        }
    }

    void octomapCallback(const octomap_msgs::Octomap::ConstPtr &msg)
    {
        ROS_INFO("Received OctoMap message");
        if (octree_)
        {
            delete octree_;
            octree_ = nullptr;
        }

        octomap::AbstractOcTree *tree = octomap_msgs::binaryMsgToMap(*msg);
        octree_ = dynamic_cast<octomap::OcTree *>(tree);

        if (!octree_)
        {
            ROS_ERROR("Failed to deserialize OctoMap!");
        }
        else
        {
            ROS_INFO("OctoMap received and deserialized successfully.");
            octomap_received_ = true;
        }
    }

    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        current_position_.setX(msg->pose.position.x);
        current_position_.setY(msg->pose.position.y);
        current_position_.setZ(msg->pose.position.z);

        current_orientation_.setX(msg->pose.orientation.x);
        current_orientation_.setY(msg->pose.orientation.y);
        current_orientation_.setZ(msg->pose.orientation.z);
        current_orientation_.setW(msg->pose.orientation.w);

        ROS_INFO("ORB-SLAM3 pose received: x=%.2f, y=%.2f, z=%.2f",
                 current_position_.x(), current_position_.y(), current_position_.z());
        pose_received_ = true;
    }

    bool isPointSafe(const tf::Vector3 &point)
    {
        if (!octree_)
        {
            ROS_WARN("No OctoMap available for safety check");
            return false;
        }

        octomap::point3d query(point.x(), point.y(), point.z());
        octomap::OcTreeNode *node = octree_->search(query);

        if (node == nullptr)
        {
            return true; // Unknown space, assumed safe
        }
        else if (!octree_->isNodeOccupied(node))
        {
            return true; // Free space
        }
        else
        {
            return false; // Occupied space
        }
    }

    tf::Vector3 findNextSafePoint(const tf::Vector3 &start, const tf::Vector3 &goal)
    {
        tf::Vector3 direction = (goal - start).normalized();
        tf::Vector3 next_point = start + direction * resolution_;

        if (isPointSafe(next_point))
        {
            return next_point;
        }
        else
        {
            ROS_WARN("Next point is not safe. Staying at current position.");
            return start;
        }
    }

    void publishGoal(const tf::Vector3 &goal_point)
    {
        geometry_msgs::PoseStamped goal_msg;
        goal_msg.header.frame_id = "world"; // Ensure consistency with OctoMap frame
        goal_msg.header.stamp = ros::Time::now();

        goal_msg.pose.position.x = goal_point.x();
        goal_msg.pose.position.y = goal_point.y();
        goal_msg.pose.position.z = goal_point.z();

        goal_msg.pose.orientation.x = current_orientation_.x();
        goal_msg.pose.orientation.y = current_orientation_.y();
        goal_msg.pose.orientation.z = current_orientation_.z();
        goal_msg.pose.orientation.w = current_orientation_.w();

        ROS_INFO("Publishing goal at (%.2f, %.2f, %.2f).", goal_point.x(), goal_point.y(), goal_point.z());
        goal_pub_.publish(goal_msg);
    }

    void run()
    {
        ros::Rate rate(10); // 10 Hz
        tf::Vector3 goal_point(goal_x_, goal_y_, goal_z_);

        while (ros::ok())
        {
            if (octomap_received_ && pose_received_)
            {
                tf::Vector3 next_point = findNextSafePoint(current_position_, goal_point);
                publishGoal(next_point);
            }
            else
            {
                ROS_INFO_THROTTLE(5, "Waiting for OctoMap and ORB-SLAM3 pose data...");
            }
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber octomap_sub_;
    ros::Subscriber pose_sub_;
    ros::Publisher goal_pub_;

    octomap::OcTree *octree_;
    double resolution_;

    tf::Vector3 current_position_;
    tf::Quaternion current_orientation_;

    bool octomap_received_;
    bool pose_received_;

    double goal_x_, goal_y_, goal_z_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "navigator_node");

    // Set the logging level to DEBUG
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug))
    {
        ros::console::notifyLoggerLevelsChanged();
    }

    Navigator navigator;
    navigator.run();

    return 0;
}
