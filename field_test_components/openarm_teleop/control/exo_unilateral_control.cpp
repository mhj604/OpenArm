#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <controller/control.hpp>
#include <controller/dynamics.hpp>
#include <openarm_port/openarm_init.hpp>
#include <robot_state.hpp>
#include <yamlloader.hpp>

std::atomic<bool> keep_running(true);
void signal_handler(int) { keep_running = false; }

class ExoRefNode : public rclcpp::Node {
public:
    ExoRefNode(std::shared_ptr<RobotSystemState> state, double scale)
        : Node("exo_openarm_reference_node"), state_(state), scale_(scale) {
        sub_ = create_subscription<sensor_msgs::msg::JointState>(
            "/exo/joint_command", 10,
            std::bind(&ExoRefNode::cb, this, std::placeholders::_1));
    }

private:
    void cb(sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->position.size() < 14) return;

        if (!zero_set_) {
            for (int i = 0; i < 7; ++i) zero_[i] = msg->position[i];
            zero_set_ = true;
            RCLCPP_WARN(get_logger(), "Captured ExoArm LEFT zero pose");
        }

        std::vector<JointState> arm_refs(7);
        for (int i = 0; i < 7; ++i) {
            arm_refs[i].position = scale_ * (msg->position[i] - zero_[i]);
            arm_refs[i].velocity = 0.0;
            arm_refs[i].effort = 0.0;
        }

        std::vector<JointState> hand_refs(1);
        hand_refs[0].position = 0.0;

        state_->arm_state().set_all_references(arm_refs);
        state_->hand_state().set_all_references(hand_refs);
    }

    std::shared_ptr<RobotSystemState> state_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
    bool zero_set_{false};
    double zero_[7]{};
    double scale_{0.2};
};

int main(int argc, char **argv) {
    std::signal(SIGINT, signal_handler);

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <urdf_path> <arm_side:left_arm|right_arm> <can_if> [scale]\n";
        return 1;
    }

    std::string urdf_path = argv[1];
    std::string arm_side = argv[2];
    std::string can_if = argv[3];
    double scale = argc >= 5 ? std::stod(argv[4]) : 0.2;

    std::string root_link = "openarm_body_link0";
    std::string leaf_link =
        arm_side == "left_arm" ? "openarm_left_hand" : "openarm_right_hand";

    YamlLoader loader("config/follower.yaml");
    auto kp = loader.get_vector("FollowerArmParam", "Kp");
    auto kd = loader.get_vector("FollowerArmParam", "Kd");
    auto Fc = loader.get_vector("FollowerArmParam", "Fc");
    auto k = loader.get_vector("FollowerArmParam", "k");
    auto Fv = loader.get_vector("FollowerArmParam", "Fv");
    auto Fo = loader.get_vector("FollowerArmParam", "Fo");

    Dynamics dynamics(urdf_path, root_link, leaf_link);
    dynamics.Init();

    auto *openarm =
        openarm_init::OpenArmInitializer::initialize_openarm(can_if, true);

    size_t arm_n = openarm->get_arm().get_motors().size();
    size_t hand_n = openarm->get_gripper().get_motors().size();

    auto state = std::make_shared<RobotSystemState>(arm_n, hand_n);

    Control control(openarm, &dynamics, &dynamics, state,
                    1.0 / FREQUENCY, ROLE_FOLLOWER,
                    arm_side, arm_n, hand_n);
    control.SetParameter(kp, kd, Fc, k, Fv, Fo);

    std::cout << "Adjusting OpenArm to initial position...\n";
    control.AdjustPosition();

    rclcpp::init(argc, argv);
    auto node = std::make_shared<ExoRefNode>(state, scale);

    std::cout << "ExoArm -> OpenArm started. scale=" << scale << "\n";

    while (rclcpp::ok() && keep_running) {
        rclcpp::spin_some(node);
        control.unilateral_step();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    openarm->disable_all();
    rclcpp::shutdown();
    return 0;
}
