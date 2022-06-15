#include <iostream>
#include <queue>
using namespace std;
/* A binary tree node has data, pointer to left child
and a pointer to right child */
struct Node
{
    int data;
    struct Node *left, *right;
    Node(int x) : data(x), left(NULL), right(NULL) {}
};
// Utility function to create a new tree node
Node *newNode(int data)
{
    Node *temp = new Node(data);
    // temp->data = data;
    temp->left = temp->right = NULL;
    return temp;
}
/* Given a binary tree, print its nodes according to the
"bottom-up" postorder traversal. */
void printPostorder(struct Node *node)
{
    if (node == NULL)
        return;
    // first recur on left subtree
    printPostorder(node->left);
    // then recur on right subtree
    printPostorder(node->right);
    // now deal with the node
    cout << node->data << " ";
}
/* Given a binary tree, print its nodes in inorder*/
void printInorder(struct Node *node)
{
    if (node == NULL)
        return;
    /* first recur on left child */
    printInorder(node->left);
    /* then print the data of node */
    cout << node->data << " ";
    /* now recur on right child */
    printInorder(node->right);
}
/* Given a binary tree, print its nodes in preorder*/
void printPreorder(struct Node *node)
{
    if (node == NULL)
        return;
    /* first print data of node */
    cout << node->data << " ";
    /* then recur on left subtree */
    printPreorder(node->left);
    /* now recur on right subtree */
    printPreorder(node->right);
}
// Iterative method to find height of Binary Tree
void printLevelOrder(Node *root)
{
    // Base Case
    if (root == NULL)
        return;

    // Create an empty queue for level order traversal
    queue<Node *> q;

    // Enqueue Root and initialize height
    q.push(root);

    while (q.empty() == false)
    {
        // Print front of queue and remove it from queue
        Node *node = q.front();
        cout << node->data << " ";
        q.pop();

        /* Enqueue left child */
        if (node->left != NULL)
            q.push(node->left);

        /*Enqueue right child */
        if (node->right != NULL)
            q.push(node->right);
    }
}
/* Driver program to test above functions*/
int main()
{
    struct Node *root = newNode(1);
    root->left = newNode(2);
    root->right = newNode(3);
    root->left->left = newNode(4);
    root->left->right = newNode(5);

    cout << "\nPreorder traversal of binary tree is \n";
    printPreorder(root);
    cout << "\nInorder traversal of binary tree is \n";
    printInorder(root);
    cout << "\nPostorder traversal of binary tree is \n";
    printPostorder(root);
    cout << "\nLeverorder traversal of binary tree is \n";
    printLevelOrder(root);
    return 0;
}
