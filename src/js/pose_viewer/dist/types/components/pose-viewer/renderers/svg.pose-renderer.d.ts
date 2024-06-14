import { PoseBodyFrameModel, PosePointModel, RGBColor } from "pose-format";
import { PoseRenderer } from "./pose-renderer";
export declare class SVGPoseRenderer extends PoseRenderer {
  renderJoint(i: number, joint: PosePointModel, color: RGBColor): any;
  renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor): any;
  render(frame: PoseBodyFrameModel): any;
}
