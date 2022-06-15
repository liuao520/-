#include <cstdio>
#include <iostream>
using namespace std;

/* Link list node */
class Node
{
public:
    int data;
    Node *next;
    Node(int x) : data(x), next(NULL) {}
};

Node *SortedMerge(Node *a, Node *b)
{
    auto dummy = new Node(-1);
    auto cur = dummy;
    while (a && b)
    {
        if (a->data < b->data)
        {
            cur->next = a;
            cur = a;
            a = a->next;
        }
        else
        {
            cur->next = b;
            cur = b;
            b = b->next;
        }
    }
    if (a)
        cur->next = a;
    else
        cur->next = b;
    return dummy->next;
}

void push(Node **head_ref, int new_data) //ͷ�巨
{
    /* allocate node */
    Node *new_node = new Node(new_data);
    /* put in the data */
    // new_node->data = new_data;
    /* link the old list off the new node */
    new_node->next = (*head_ref);
    /* move the head to point to the new node */
    (*head_ref) = new_node;
}

/* Function to print nodes in a given linked list */
void printList(Node *node)
{
    while (node != NULL)
    {
        cout << node->data << " ";
        node = node->next;
    }
    cout << endl;
}

Node *reverse(Node *head)
{
    // Initialize current, previous and
    // next pointers
    Node *current = head;
    Node *prev = NULL;
    while (current != NULL)
    {
        // Store next
        auto next = current->next;
        // Reverse current node's pointer
        current->next = prev;
        // Move pointers one position ahead.
        prev = current;
        current = next;
    }
    return prev;
}
Node *rever_list(Node *head)
{
    if (!head || !head->next)
        return head;
    auto tail = rever_list(head->next);
    head->next->next = head;
    head->next = nullptr;
    return tail;
}

int main()
{
    /* Start with the empty list */
    Node *res = NULL;
    Node *a = NULL;
    Node *b = NULL;

    /* Let us create two sorted linked lists*/
    push(&a, 15);
    push(&a, 10);
    push(&a, 5);

    push(&b, 20);
    push(&b, 3);
    push(&b, 2);

    printList(a);
    // reverse(a);
    res = rever_list(a);

    printList(res);

    return 0;
}
