#include "BasicScene.h"
#include <Eigen/src/Core/Matrix.h>
#include <edges.h>
#include <memory>
#include <per_face_normals.h>
#include <read_triangle_mesh.h>
#include <utility>
#include <vector>
#include "GLFW/glfw3.h"
#include "Mesh.h"
#include "PickVisitor.h"
#include "Renderer.h"
#include "ObjLoader.h"
#include "IglMeshLoader.h"

#include "igl/per_vertex_normals.h"
#include "igl/per_face_normals.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/edge_flaps.h"
#include "igl/loop.h"
#include "igl/upsample.h"
#include "igl/AABB.h"
#include "igl/parallel_for.h"
#include "igl/shortest_edge_and_midpoint.h"
#include "igl/circulation.h"
#include "igl/edge_midpoints.h"
#include "igl/collapse_edge.h"
#include "igl/edge_collapse_is_valid.h"
#include "igl/write_triangle_mesh.h"

#include "Utility.h"

using namespace cg3d;

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    // create backround and root
    numberOfCyl = 3;
    lastLinkIndex = numberOfCyl - 1;
    AnimateCCD = false;
    camera = Camera::Create("camera", fov, float(width) / height, near, far);
    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    camera->Translate(30, Axis::Z);
    root->RotateByDegree(90, Axis::X);
    auto daylight{ std::make_shared<Material>("daylight", "shaders/cubemapShader") };
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{ Model::Create("background", Mesh::Cube(), daylight) };
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();

    // create backround
    auto program = std::make_shared<Program>("shaders/phongShader");
    auto program1 = std::make_shared<Program>("shaders/pickingShader");

    // create materials
    auto material{ std::make_shared<Material>("material", program) }; // empty material
    auto material1{ std::make_shared<Material>("material", program1) }; // empty material
    material->AddTexture(0, "textures/box0.bmp", 2);

    // create meshes
    auto sphereMesh{ IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj") };
    auto cylMesh{ IglLoader::MeshFromFiles("cyl_igl","data/zcylinder.obj") };
    auto cubeMesh{ IglLoader::MeshFromFiles("cube_igl","data/cube_old.obj") };
    sphere1 = Model::Create("sphere", sphereMesh, material);
    cube = Model::Create("cube", cubeMesh, material);

    // create axis
    Eigen::MatrixXd vertices(6, 3);
    vertices << -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1;
    Eigen::MatrixXi faces(3, 2);
    faces << 0, 1, 2, 3, 4, 5;
    Eigen::MatrixXd vertexNormals = Eigen::MatrixXd::Ones(6, 3);
    Eigen::MatrixXd textureCoords = Eigen::MatrixXd::Ones(6, 2);
    std::shared_ptr<Mesh> coordsys = std::make_shared<Mesh>("coordsys", vertices, faces, vertexNormals, textureCoords);
    axis.push_back(Model::Create("axis", coordsys, material1));
    axis[0]->mode = 1;
    axis[0]->Scale(4, Axis::XYZ);
    
    // add axis[0] to root
    root->AddChild(axis[0]);
    float scaleFactor = 1;

    // create first cylinder and add as child to root
    cyls.push_back(Model::Create("cyl", cylMesh, material));
    cyls[0]->Scale(scaleFactor, Axis::Z);
    cyls[0]->SetCenter(Eigen::Vector3f(0,0,-0.8f*scaleFactor));
    cyls[0]->RotateByDegree(90, Eigen::Vector3f(0,1,0));
    root->AddChild(cyls[0]);
    
    // create next cylinders
    for (int i = 1; i < numberOfCyl; i++)
    {
        // cylinders
        cyls.push_back(Model::Create("cyl", cylMesh, material));
        cyls[i]->Scale(scaleFactor, Axis::Z);
        cyls[i]->Translate(1.6f * scaleFactor, Axis::Z);
        cyls[i]->SetCenter(Eigen::Vector3f(0,0,-0.8f*scaleFactor));
        cyls[i - 1]->AddChild(cyls[i]);

        // add axis
        axis.push_back(Model::Create("axis", coordsys, material1));
        axis[i]->mode = 1;
        axis[i]->Scale(4, Axis::XYZ);
        axis[i]->Translate(0.8f* scaleFactor,Axis::Z);
        
        // add axis to cylinder
        cyls[i - 1]->AddChild(axis[i]);
    }
    cyls[0]->Translate({0,0,0.8f*scaleFactor});

    auto morphFunc = [](Model* model, cg3d::Visitor* visitor) {
        return model->meshIndex;
    };

    // create cube
    autoCube = AutoMorphingModel::Create(*cube, morphFunc);
    autoCube->Translate({ -4.8,0,0 });
    autoCube->Scale(1.5f);

    autoCube->showWireframe = true;
    sphere1->showWireframe = true;
    root->AddChild(sphere1);
    root->AddChild(autoCube);
    cube->mode = 1;
}


void BasicScene::ikCyclicCoordinateDecentMethod(std::shared_ptr<AutoMorphingModel> target) {
    if (AnimateCCD) {
        // get cube location
        Eigen::Vector3f targetDestination = target->GetAggregatedTransform().block<3, 1>(0, 3);
        // get first link base
        Eigen::Vector3f firstLinkBase = getBaseOfCyl(0);
        // check if target isnt far enough
        if ((targetDestination - firstLinkBase).norm() > CYL_LENGTH * numberOfCyl) { 
            std::cout << "cannot reach" << std::endl;
            AnimateCCD = false;
            return;
        }
        int currIndex = lastLinkIndex;
        while (currIndex != -1) {
            std::shared_ptr<Model> currLink = cyls[currIndex];
            Eigen::Vector3f R = getBaseOfCyl(currIndex);
            Eigen::Vector3f E = getTipOfCyl(lastLinkIndex);
            Eigen::Vector3f RD = targetDestination - R;
            Eigen::Vector3f RE = E - R;
            Eigen::Vector3f normal = RE.normalized().cross(RD.normalized()); //calculates plane normal
            float distance = (targetDestination - E).norm();
            if (distance < 0.05) {
                std::cout << "destination reached, distance: " << distance << std::endl;
                AnimateCCD = false;
                return;
            }
            float dotProduct = RD.normalized().dot(RE.normalized());
            // make sure dotProduct is between -1 and 1
            if (dotProduct > 1)
                dotProduct = 1;
            else if (dotProduct < -1)
                dotProduct = -1;
            float angle = acos(dotProduct) / 50;
            // rotation vector
            Eigen::Vector3f rotationVector = (currLink->GetAggregatedTransform()).block<3, 3>(0, 0).inverse() * normal; 
            if (currIndex != 0) {
                // rotate linke
                currLink->Rotate(angle, rotationVector);
                // get position after the rotation
                E = getTipOfCyl(lastLinkIndex); 
                RE = E - R;
                Eigen::Vector3f RParent = getBaseOfCyl(currIndex - 1);
                RD = RParent - R;
                // find angle between parent and link
                double constraint = 0.5;
                double parentDot = RD.normalized().dot(RE.normalized());
                // make sure parentDot is between -1 and 1
                if (parentDot > 1)
                    parentDot = 1;
                if (parentDot < -1)
                    parentDot = 1;
                double parentAngle = acos(parentDot);
                // rotate back
                currLink->Rotate(-angle, rotationVector);
                // fix angle
                if (parentAngle < constraint) {
                    angle = angle - (constraint - parentAngle);
                }

            }
            currLink->Rotate(angle, rotationVector);
            currIndex -= 1;
        }
    }

}

Eigen::Vector3f BasicScene::getTipOfCyl(int cyl_index) {
    Eigen::Matrix4f linkTransform = cyls[cyl_index]->GetAggregatedTransform();
    Eigen::Vector3f linkCenter = Eigen::Vector3f(linkTransform.col(3).x(), linkTransform.col(3).y(), linkTransform.col(3).z());
    return linkCenter + cyls[cyl_index]->GetRotation() * Eigen::Vector3f(0 , 0, CYL_LENGTH / 2);
}

Eigen::Vector3f BasicScene::getBaseOfCyl(int cyl_index) {
    // get base of cylinder
    Eigen::Matrix4f linkTransform = cyls[cyl_index]->GetAggregatedTransform();
    Eigen::Vector3f linkCenter = Eigen::Vector3f(linkTransform.col(3).x(), linkTransform.col(3).y(), linkTransform.col(3).z());
    return linkCenter - cyls[cyl_index]->GetRotation() * Eigen::Vector3f(0, 0, CYL_LENGTH / 2);
}


void BasicScene::Update(const Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model)
{
    Scene::Update(program, proj, view, model);
    program.SetUniform4f("lightColor", 0.8f, 0.3f, 0.0f, 0.5f);
    program.SetUniform4f("Kai", 1.0f, 0.3f, 0.6f, 1.0f);
    program.SetUniform4f("Kdi", 0.5f, 0.5f, 0.0f, 1.0f);
    program.SetUniform1f("specular_exponent", 5.0f);
    program.SetUniform4f("light_position", 0.0, 15.0f, 0.0, 1.0f);
    //    cyl->Rotate(0.001f, Axis::Y);
    ikCyclicCoordinateDecentMethod(autoCube);
}

void BasicScene::MouseCallback(Viewport* viewport, int x, int y, int button, int action, int mods, int buttonState[])
{
    // default mouse button press behavior
    if (action == GLFW_PRESS) { 
        PickVisitor visitor;
        visitor.Init();
        renderer->RenderViewportAtPos(x, y, &visitor); 
        auto modelAndDepth = visitor.PickAtPos(x, renderer->GetWindowHeight() - y);
        // draw again to avoid flickering
        renderer->RenderViewportAtPos(x, y); 
        pickedModel = modelAndDepth.first ? std::dynamic_pointer_cast<Model>(modelAndDepth.first->shared_from_this()) : nullptr;
        pickedModelDepth = modelAndDepth.second;
        camera->GetRotation().transpose();
        xAtPress = x;
        yAtPress = y;

        // for non-pickable models we need only pickedModelDepth for mouse movement calculations later
        if (pickedModel)
            if (!pickedModel->isPickable)
                pickedModel = nullptr;
            else
                pickedToutAtPress = pickedModel->GetTout();
        else
            cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::ScrollCallback(Viewport* viewport, int x, int y, int xoffset, int yoffset, bool dragging, int buttonState[])
{
    auto system = camera->GetRotation().transpose();
    if (pickedModel) {
        std::shared_ptr<cg3d::Model> currModel = pickedModel;
        
        if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end()) 
            currModel = cyls[0];
        currModel->TranslateInSystem(system, { 0, 0, -float(yoffset) });
        pickedToutAtPress = currModel->GetTout();
    }
    else {
        camera->TranslateInSystem(system, { 0, 0, -float(yoffset) });
        cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::CursorPosCallback(Viewport* viewport, int x, int y, bool dragging, int* buttonState)
{
    if (dragging) {
        auto system = camera->GetRotation().transpose() * GetRotation();
        auto moveCoeff = camera->CalcMoveCoeff(pickedModelDepth, viewport->width);
        auto angleCoeff = camera->CalcAngleCoeff(viewport->width);
        if (pickedModel) {
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE) {
                std::shared_ptr<cg3d::Model> currModel = pickedModel;
                // if picked item is cylinder make sure the arm doesn't break
                if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end())
                    currModel = cyls[0];
                currModel->TranslateInSystem(system, { -float(xAtPress - x) / moveCoeff, 0, float(y - yAtPress) / moveCoeff });
            }
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Y);
                pickedModel->RotateInSystem(system, float(yAtPress - y) / angleCoeff, Axis::X);
            }
        }
        else {
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                root->TranslateInSystem(system, { -float(xAtPress - x) / moveCoeff / 10.0f, float(yAtPress - y) / moveCoeff / 10.0f, 0 });
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                root->RotateInSystem(system, float(x - xAtPress) / 180.0f, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                root->RotateInSystem(system, float(x - xAtPress) / angleCoeff, Axis::Y);
                root->RotateInSystem(system, float(y - yAtPress) / angleCoeff, Axis::X);
            }
        }
        xAtPress = x;
        yAtPress = y;
    }
}

void BasicScene::KeyCallback(Viewport* viewport, int x, int y, int key, int scancode, int action, int mods)
{
    auto system = camera->GetRotation().transpose();

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key)
        {
        case GLFW_KEY_SPACE:
            AnimateCCD = !AnimateCCD;
            break;
        case GLFW_KEY_P: {
            Eigen::Matrix3f rotation;
            if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end())
                rotation = pickedModel->GetRotation();
            else
                rotation = root->GetRotation();
            Eigen::Vector3f eulerAngles = rotation.eulerAngles(2, 0, 2); // get zxz
            // first z rotation
            Eigen::Matrix3f phi; 
            phi.row(0) = Eigen::Vector3f(cos(eulerAngles.x()), -sin(eulerAngles.x()), 0);
            phi.row(1) = Eigen::Vector3f(sin(eulerAngles.x()), cos(eulerAngles.x()), 0);
            phi.row(2) = Eigen::Vector3f(0, 0, 1);

            // x rotation
            Eigen::Matrix3f theta; 
            theta.row(0) = Eigen::Vector3f(1, 0, 0);
            theta.row(1) = Eigen::Vector3f(0, cos(eulerAngles.y()), -sin(eulerAngles.y()));
            theta.row(2) = Eigen::Vector3f(0, sin(eulerAngles.y()), cos(eulerAngles.y()));

            // second z rotation
            Eigen::Matrix3f psi; 
            psi.row(0) = Eigen::Vector3f(cos(eulerAngles.z()), -sin(eulerAngles.z()), 0);
            psi.row(1) = Eigen::Vector3f(sin(eulerAngles.z()), cos(eulerAngles.z()), 0);
            psi.row(2) = Eigen::Vector3f(0, 0, 1);

            std::cout << "Phi matrix: " << std::endl << phi << std::endl;
            std::cout << "Theta matrix: " << std::endl << theta << std::endl;
            std::cout << "Psi matrix: " << std::endl << psi << std::endl;
            break;
        }
        case GLFW_KEY_T: {
            Eigen::Vector3f tip = getTipOfCyl(lastLinkIndex);
            std::cout << "target = (x: " << tip.x() << " y: " << tip.y() << " z: " << tip.z() << ")" << std::endl; }
            break;
        case GLFW_KEY_N: {
            if (pickedIndex < cyls.size() - 1)
                pickedIndex++;
            else
                pickedIndex = 0; 
            pickedModel = cyls[pickedIndex];
        }
            break;
        case GLFW_KEY_LEFT:
            if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end()) {
                cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::Y);
            }
            else
                root->RotateInSystem(system, 0.1f, Axis::Y);
            break;
        case GLFW_KEY_RIGHT:
            if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end()) {
                cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::Y);
            }
            else
                root->RotateInSystem(system, -0.1f, Axis::Y);
            break;
        case GLFW_KEY_UP:
            if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end()) {
                cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::X);
            }
            else
                root->RotateInSystem(system, 0.1f, Axis::X);
            break;
        case GLFW_KEY_DOWN:
            if (std::find(cyls.begin(), cyls.end(), pickedModel) != cyls.end()) {
                cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::X);
            }
            else
                root->RotateInSystem(system, -0.1f, Axis::X);
            break;
        }
    }
}